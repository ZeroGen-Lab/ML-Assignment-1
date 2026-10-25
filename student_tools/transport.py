"""Standard-library HTTPS transport for the four student operations.

make_request(base_url, *, get_headers=None, timeout=30) returns the
request-callable protocol expected by student_tools.client. This module does
not read identity from the environment: without an injected get_headers
provider the callable fails closed with SERVICE_NOT_CONFIGURED before any
network activity. The explicit self_report factory supplies self-reported
metadata, not proof of identity. TLS verification and hostname checks are
always on; there is no parameter to disable them.
No redirects are followed, no cookies are kept, and each call is a single
attempt: retries are the caller's policy with the client idempotency key.
"""
import base64
import http.client
import ipaddress
import json
import re
import ssl
import urllib.parse

from student_tools.client import ClientError, ERRORS, MAX_ARCHIVE_BYTES, canonical_selected_tasks

MAX_RESPONSE_BYTES = 1024 * 1024
MAX_DEPTH = 16
_DEFAULT_PORT = 443
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_HOST = re.compile(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?'
                   r'(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$')
# Transport-owned framing headers and the LEGACY self-reported header names
# can never arrive through the channel callback. The explicit self-report
# mode (docs/codex-identity-decision-20261010.md) is the approved exception:
# its distinct X-Self-Reported-Id/-Name/-Nickname prefix is supplied by
# student_tools.self_report and is deliberately not forbidden here; other
# names stay available for a future approved trusted-channel mechanism.
_FORBIDDEN_HEADERS = frozenset({'host', 'content-length', 'content-encoding',
                                'transfer-encoding', 'content-type', 'cookie',
                                'x-student-id', 'x-student-name', 'x-student-nickname'})
_PATHS = {'submit': '/api/student/submit', 'status': '/api/student/status',
          'list_submissions': '/api/student/submissions',
          'set_nickname': '/api/student/nickname'}
_FIELDS = {'submit': frozenset({'idempotency_key', 'ai_declaration'}),
           'status': frozenset({'submission_id'}),
           'list_submissions': frozenset({'limit', 'offset'}),
           'set_nickname': frozenset({'nickname'})}


def _parse_origin(base_url):
    """Parse an exact http:// or https:// origin.

    Returns (host, port, secure); https keeps the exact previous validation
    and the 443 default, http uses port 80 by default and keeps every other
    rule (no query/fragment/userinfo/path, legal host, port range) unchanged.
    """
    if type(base_url) is not str:
        raise ValueError('base_url must be an exact origin string (http or https)')
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in ('http', 'https') or parts.query or parts.fragment:
        raise ValueError('base_url must be an exact origin (http or https)')
    if parts.username is not None or parts.password is not None:
        raise ValueError('base_url must not carry userinfo')
    if parts.path not in ('', '/'):
        raise ValueError('base_url must not carry a path')
    host = parts.hostname
    if not host or len(host) > 253:
        raise ValueError('base_url has no usable host')
    port = parts.port  # raises ValueError itself on an illegal port token
    secure = parts.scheme == 'https'
    if port is None:
        port = _DEFAULT_PORT if secure else 80
    if not 1 <= port <= 65535:
        raise ValueError('base_url port out of range')
    if _HOST.fullmatch(host) is None:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            raise ValueError('base_url host is not a valid name or address') from None
    return host, port, secure


def _gate(operation, fields, payload):
    keys = _FIELDS.get(operation)
    if operation == 'submit' and type(fields) is dict and 'selected_tasks' in fields:
        keys = keys | {'selected_tasks'}
    if keys is None or type(fields) is not dict or set(fields) != keys:
        raise ClientError('INVALID_REQUEST')
    if operation == 'list_submissions':
        if type(fields['limit']) is not int or type(fields['offset']) is not int:
            raise ClientError('INVALID_REQUEST')
    elif operation == 'submit':
        if not all(type(fields[key]) is str for key in ('idempotency_key', 'ai_declaration')):
            raise ClientError('INVALID_REQUEST')
        if 'selected_tasks' in fields:
            tasks = fields['selected_tasks']
            if type(tasks) is not list or tasks != list(canonical_selected_tasks(tasks)):
                raise ClientError('INVALID_REQUEST')
    elif not all(type(value) is str for value in fields.values()):
        raise ClientError('INVALID_REQUEST')
    if payload is not None and operation != 'submit':
        raise ClientError('INVALID_REQUEST')
    if operation == 'submit':
        if type(payload) is not bytes or not 0 < len(payload) <= MAX_ARCHIVE_BYTES:
            raise ClientError('INVALID_REQUEST')


def _channel_headers(provided):
    if type(provided) is not dict:
        raise ClientError('SERVICE_NOT_CONFIGURED')
    headers = {}
    for name, value in provided.items():
        if (type(name) is not str or type(value) is not str
                or name.lower() in _FORBIDDEN_HEADERS
                or _HEADER_NAME.fullmatch(name) is None
                or not value.isascii()
                or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)):
            raise ClientError('SERVICE_NOT_CONFIGURED')
        headers[name] = value
    return headers


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ClientError('INVALID_RESPONSE')
        result[key] = value
    return result


def _no_constant(_name):
    raise ClientError('INVALID_RESPONSE')


def _scan_depth(raw):
    depth, in_string, escaped = 0, False, False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            if depth > MAX_DEPTH:
                raise ClientError('INVALID_RESPONSE')
        elif byte in (0x7D, 0x5D):
            depth -= 1


def _parse(raw):
    _scan_depth(raw)
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=_no_duplicates,
                          parse_constant=_no_constant)
    except ClientError:
        raise
    except (UnicodeDecodeError, ValueError):
        raise ClientError('INVALID_RESPONSE') from None


def _envelope(document):
    if type(document) is not dict or type(document.get('ok')) is not bool:
        raise ClientError('INVALID_RESPONSE')
    if document['ok'] is False:
        error = document.get('error')
        if (set(document) != {'ok', 'error'} or type(error) is not dict
                or set(error) != {'code'}):
            raise ClientError('INVALID_RESPONSE')
        code = error['code']
        if type(code) is not str or code not in ERRORS:
            raise ClientError('INVALID_RESPONSE')
    return document


def _read_bounded(response):
    total, chunks = 0, []
    while True:
        chunk = response.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise ClientError('INVALID_RESPONSE')
        chunks.append(chunk)
    return b''.join(chunks)


def make_request(base_url, *, get_headers=None, timeout=30):
    """Return the request callable for student_tools.client wiring.

    get_headers supplies the explicitly configured channel metadata; it may
    come from the approved self_report factory and is not itself proof of
    person. It is called once per request before any connection is opened,
    so a missing or broken channel fails closed with zero network activity.
    """
    host, port, secure = _parse_origin(base_url)

    def request(operation, fields, *, payload=None):
        if get_headers is None:
            raise ClientError('SERVICE_NOT_CONFIGURED')
        _gate(operation, fields, payload)
        try:
            provided = get_headers()
        except Exception:
            raise ClientError('SERVICE_NOT_CONFIGURED') from None
        headers = _channel_headers(provided)
        headers['Content-Type'] = 'application/json'
        document = dict(fields)
        if operation == 'submit':
            document['payload_b64'] = base64.b64encode(payload).decode('ascii')
        body = json.dumps(document, separators=(',', ':')).encode('utf-8')
        try:
            if port is None:
                pass
            if secure:
                connection = http.client.HTTPSConnection(
                    host, port, timeout=timeout,
                    context=ssl.create_default_context())
            else:
                connection = http.client.HTTPConnection(host, port,
                                                        timeout=timeout)
            try:
                connection.request('POST', _PATHS[operation], body=body, headers=headers)
                response = connection.getresponse()
                raw = _read_bounded(response)
            finally:
                connection.close()
        except ClientError:
            raise
        except Exception:
            raise ClientError('SERVICE_UNAVAILABLE') from None
        return _envelope(_parse(raw))

    return request
