"""Internal protocol-only client; no network/authentication or CLI configured.

An injected request callable is not evidence of service identity. Deployment
must separately establish endpoint origin and authorization. No export values
or host context are read, synthesized or transmitted here.
"""
import copy
import re
import secrets
import unicodedata
from datetime import datetime, timezone

from student_tools.package import MAX_TOTAL_BYTES


AI_VALUES = frozenset({'independent', 'ai_assisted'})
STATES = frozenset({'QUEUED', 'RUNNING', 'COMPLETED', 'STUDENT_FAILURE', 'INFRASTRUCTURE_FAILURE'})
ERRORS = frozenset({'INVALID_REQUEST', 'AUTHORIZATION_REQUIRED', 'AUTHORIZATION_CHANGED',
                   'NOT_FOUND', 'CONFLICT', 'DEADLINE_PASSED', 'PACKAGE_REJECTED', 'SERVICE_UNAVAILABLE',
                   'DAILY_SUBMISSION_LIMIT', 'PENDING_SUBMISSION_LIMIT'})
COMMON = {'submission_id', 'received_at', 'ai_declaration', 'data_version', 'eval_config_id'}
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES
TASK_ORDER = ('lr', 'svm', 'gbdt', 'kmeans', 'hmm')
TASKS = frozenset(TASK_ORDER)
_SELECTION_OMITTED = object()
SLOT_STATES = frozenset(('measured', 'student_failure', 'infrastructure_failure', 'not_run'))
STUDENT_SLOT_FAILURES = frozenset(('student_runtime', 'student_timeout', 'student_invalid_output'))
OUTER_FAILURES = frozenset(('student_compile', 'student_tasks', 'infrastructure'))


class ClientError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code='INVALID_RESPONSE'):
    if not condition:
        raise ClientError(code)


def canonical_selected_tasks(value, *, code='INVALID_REQUEST'):
    """Student-side copy of the ledger's closed four/five-task contract."""
    _require(isinstance(value, (list, tuple)) and len(value) in (4, 5), code)
    _require(all(isinstance(task, str) and task in TASKS for task in value), code)
    _require(len(set(value)) == len(value), code)
    return tuple(task for task in TASK_ORDER if task in value)


def _text(value, maximum):
    return (type(value) is str and 0 < len(value) <= maximum and bool(value.strip())
            and not any(unicodedata.category(c).startswith('C')
                        or unicodedata.category(c) in ('Zl', 'Zp') for c in value))


def _id(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{32}', value) is not None


def _keys(value, keys):
    _require(type(value) is dict and all(type(key) is str for key in value) and set(value) == set(keys))


def _utc(value):
    if type(value) is not str or re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00', value) is None:
        return False
    try:
        instant = datetime.fromisoformat(value)
        return instant.astimezone(timezone.utc).isoformat(timespec='microseconds') == value
    except (ValueError, OverflowError):
        return False


def _summary(result, status):
    """Strict result-summary validation: no score, case, metric, hash,
    environment, timing or extra field may ride on the wire, and the
    summary must agree with the parent ledger status."""
    if result['available'] is False:
        _keys(result, {'available', 'code'})
        _require(type(result['code']) is str and result['code'] == 'RESULT_NOT_AVAILABLE')
        return
    _require(result['available'] is True)
    _keys(result, {'available', 'failure_type', 'tasks'})
    failure, tasks = result['failure_type'], result['tasks']
    _require(failure is None or (type(failure) is str and failure in OUTER_FAILURES))
    _require(type(tasks) is dict and all(type(key) is str for key in tasks) and set(tasks) == TASKS)
    statuses = []
    for task in TASKS:
        slot = tasks[task]
        _keys(slot, {'status', 'failure_type'})
        state, slot_failure = slot['status'], slot['failure_type']
        _require(type(state) is str and state in SLOT_STATES)
        if state in ('measured', 'not_run'):
            _require(slot_failure is None)
        elif state == 'infrastructure_failure':
            _require(type(slot_failure) is str and slot_failure == 'infrastructure')
        else:
            _require(type(slot_failure) is str and slot_failure in STUDENT_SLOT_FAILURES)
        statuses.append(state)
    if status == 'COMPLETED':
        _require(failure is None and all(state == 'measured' for state in statuses))
    elif status == 'STUDENT_FAILURE':
        if failure == 'student_compile':
            _require(all(state == 'not_run' for state in statuses))
        elif failure == 'student_tasks':
            _require(all(state in ('measured', 'student_failure') for state in statuses)
                     and 'student_failure' in statuses)
        else:
            _require(False)
    else:  # INFRASTRUCTURE_FAILURE with a summary body.
        _require(type(failure) is str and failure == 'infrastructure'
                 and any(state in ('infrastructure_failure', 'not_run') for state in statuses))


def _submission(data, *, receipt=False):
    keys = COMMON | {'kind', 'status', 'attempt'}
    if type(data) is dict and 'selected_tasks' in data:
        keys |= {'selected_tasks'}
        tasks = data['selected_tasks']
        _require(type(tasks) is list)
        _require(tasks == list(canonical_selected_tasks(tasks, code='INVALID_RESPONSE')))
    _keys(data, keys if receipt else keys | {'result'})
    _require(_id(data['submission_id']) and _utc(data['received_at']))
    _require(type(data['ai_declaration']) is str and data['ai_declaration'] in AI_VALUES)
    _require(all(_text(data[k], 128) for k in ('data_version', 'eval_config_id')))
    _require(type(data['attempt']) is int and data['attempt'] >= 0)
    if receipt:
        _require(type(data['kind']) is str and data['kind'] == 'submission_receipt'
                 and type(data['status']) is str and data['status'] == 'QUEUED' and data['attempt'] == 0)
    else:
        _require(type(data['kind']) is str and data['kind'] == 'submission_status'
                 and type(data['status']) is str and data['status'] in STATES)
        _summary(data['result'], data['status'])


def _response(operation, fields, response):
    _require(type(response) is dict and type(response.get('ok')) is bool)
    if not response['ok']:
        _keys(response, {'ok', 'error'})
        _keys(response['error'], {'code'})
        code = response['error']['code']
        _require(type(code) is str and code in ERRORS)
        raise ClientError(code)
    _keys(response, {'ok', 'data'})
    data = response['data']
    if operation == 'submit':
        _submission(data, receipt=True)
        _require(data['ai_declaration'] == fields['ai_declaration'])
        _require(data.get('selected_tasks') == fields.get('selected_tasks'))
    elif operation == 'status':
        _submission(data)
        _require(data['submission_id'] == fields['submission_id'])
    elif operation == 'list_submissions':
        _keys(data, {'kind', 'items', 'limit', 'offset'})
        _require(type(data['kind']) is str and data['kind'] == 'submission_list')
        _require(type(data['limit']) is int and type(data['offset']) is int
                 and data['limit'] == fields['limit'] and data['offset'] == fields['offset'])
        _require(type(data['items']) is list and len(data['items']) <= fields['limit'])
        for item in data['items']:
            _submission(item)
        identifiers = [item['submission_id'] for item in data['items']]
        _require(len(set(identifiers)) == len(identifiers))
        order = [(item['received_at'], item['submission_id']) for item in data['items']]
        _require(order == sorted(order, reverse=True))
    else:  # Only set_nickname reaches this private helper.
        _keys(data, {'kind', 'nickname'})
        _require(type(data['kind']) is str and data['kind'] == 'nickname_updated' and _text(data['nickname'], 80)
                 and data['nickname'] == fields['nickname'])
    return copy.deepcopy(data)


def _call(request, operation, fields, *, payload=None):
    if request is None:
        raise ClientError('SERVICE_NOT_CONFIGURED')
    _require(callable(request), 'SERVICE_NOT_CONFIGURED')
    # Copy fields so a transport cannot rewrite expected response bindings.
    try:
        response = request(operation, copy.deepcopy(fields), payload=payload)
    except ClientError as error:
        # A channel-side refusal with a known protocol code keeps its code:
        # the wire protocol codes plus SERVICE_NOT_CONFIGURED, which _call
        # itself raises for a missing request. Anything else (unknown codes,
        # unknown exception types) stays a fixed SERVICE_UNAVAILABLE, unechoed.
        if type(error.code) is str and error.code in ERRORS | {'SERVICE_NOT_CONFIGURED'}:
            raise
        raise ClientError('SERVICE_UNAVAILABLE') from None
    except Exception:
        raise ClientError('SERVICE_UNAVAILABLE') from None
    try:
        return _response(operation, fields, response)
    except ClientError:
        raise
    except Exception:
        raise ClientError('INVALID_RESPONSE') from None


def new_submission_key():
    """Generate once per intent. Caller retains it for all retries."""
    return secrets.token_hex(16)


def submit(*, idempotency_key, payload, ai_declaration,
           selected_tasks=_SELECTION_OMITTED, request=None):
    _require(_text(idempotency_key, 128) and type(ai_declaration) is str and ai_declaration in AI_VALUES
             and type(payload) is bytes and 0 < len(payload) <= MAX_ARCHIVE_BYTES, 'INVALID_REQUEST')
    fields = {'idempotency_key': idempotency_key, 'ai_declaration': ai_declaration}
    if selected_tasks is not _SELECTION_OMITTED:
        fields['selected_tasks'] = list(canonical_selected_tasks(selected_tasks))
    return _call(request, 'submit', fields, payload=payload)


def status(submission_id, *, request=None):
    _require(_id(submission_id), 'INVALID_REQUEST')
    return _call(request, 'status', {'submission_id': submission_id})


def list_submissions(*, limit=20, offset=0, request=None):
    _require(type(limit) is int and 1 <= limit <= 100 and type(offset) is int
             and 0 <= offset <= 1000000, 'INVALID_REQUEST')
    return _call(request, 'list_submissions', {'limit': limit, 'offset': offset})


def set_nickname(nickname, *, request=None):
    _require(_text(nickname, 80), 'INVALID_REQUEST')
    return _call(request, 'set_nickname', {'nickname': nickname})
