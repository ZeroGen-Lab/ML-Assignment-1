"""Idempotent submission-intent persistence. Local storage only, no network.

Contract (docs/student-intent-recovery-20261009.md):

- ``create_intent(root, *, payload, ai_declaration)`` is an EXPLICIT new
  intent: it always generates a fresh key, even for byte-identical payload
  and the same AI declaration. The exact payload bytes, their digest, the AI
  declaration and the key are persisted BEFORE any network call could happen
  (this module makes none and takes no request callable). The payload must
  pass ``package.validate_package`` (structure only; nothing is extracted,
  compiled or executed).
- ``read_intent(root, key)`` restores the original triple: the exact stored
  ZIP bytes, the stored AI declaration and the key. Re-packaging the project
  yields different bytes and therefore requires a NEW intent; a stored key is
  never re-pointed at new bytes.
- ``record_receipt(root, key, *, submission_id, received_at)`` writes a
  closed-schema receipt exactly once: identical replay is a safe no-op, a
  different receipt is refused, history is never overwritten. A receipt is
  bookkeeping, NOT authentication.

No identity strings, headers, tokens or credentials are ever stored, and no
authoritative identity is cached. Layout: ``root/<key>/`` with 0700
directories and 0600 files, exclusive creation and fsync; reads reject
symlinks, non-regular files, wrong owner/mode, missing, duplicated, tampered,
oversized or unknown structure (an intent directory must contain exactly
``source.zip`` and ``manifest.json`` plus at most ``receipt.json``).
Cumulative quotas (256 intents, 256MiB total, candidate caps) are enforced
under an exclusive ``flock`` on the root directory so concurrent creates
coordinate atomically instead of check-then-write racing; a pre-existing root
path is verified and NEVER re-chmodded (symlinked or wrong-mode roots are
rejected with their target permissions untouched). Platforms lacking
``flock``/``O_NOFOLLOW``/``O_DIRECTORY``/``geteuid`` fail closed with a
stable IntentError (Windows semantics are NOT verified, and Windows is not
supported); this is not a full TOCTOU/hard-link/power-loss guarantee.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import stat
import zlib

try:
    import fcntl
except ImportError:               # non-POSIX platform: fail closed, never lockless
    fcntl = None

from student_tools import client, package

SOURCE = 'source.zip'
MANIFEST = 'manifest.json'
RECEIPT = 'receipt.json'
INTENT_KIND = 'student_intent'
RECEIPT_KIND = 'submission_receipt'
MANIFEST_KEYS = frozenset(('kind', 'idempotency_key', 'source_sha256',
                           'ai_declaration'))
RECEIPT_KEYS = frozenset(('kind', 'submission_id', 'received_at'))
MAX_MANIFEST_BYTES = 512
MAX_RECEIPT_BYTES = 256
MAX_INTENTS = 256
MAX_TOTAL_INTENT_BYTES = 256 * 1024 * 1024
_CHUNK = 65536
_CLOEXEC = getattr(os, 'O_CLOEXEC', 0)
_NOFOLLOW = getattr(os, 'O_NOFOLLOW', None)
_DIRECTORY = getattr(os, 'O_DIRECTORY', None)
_INVALID = 'invalid or unreadable student intent'


class IntentError(ValueError):
    """Stable boundary rejection; never carries private paths or content."""


def _posix_primitives():
    """Fail closed on platforms lacking the primitives this module relies on."""
    if (fcntl is None or _NOFOLLOW is None or _DIRECTORY is None
            or not callable(getattr(os, 'geteuid', None))):
        raise IntentError(_INVALID)


@dataclass(frozen=True)
class Receipt:
    submission_id: str
    received_at: str


@dataclass(frozen=True)
class IntentRecord:
    idempotency_key: str
    source_sha256: str
    ai_declaration: str
    payload: bytes
    receipt: object = None    # None or Receipt
    selected_tasks: object = None    # None only for legacy manifests


def _key(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{32}', value) is None:
        raise IntentError(_INVALID)
    return value


def _utc(value):
    """Strict UTC microsecond ISO-8601, mirroring the client wire shape."""
    if (type(value) is not str
            or re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00',
                            value) is None):
        raise IntentError(_INVALID)
    try:
        instant = datetime.fromisoformat(value)
    except (ValueError, OverflowError):
        raise IntentError(_INVALID) from None
    if instant.astimezone(timezone.utc).isoformat(
            timespec='microseconds') != value:
        raise IntentError(_INVALID)


def _dir_ok(info):
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise IntentError(_INVALID)


def _open_root(root, *, create):
    value = os.fspath(root)
    if (type(value) is not str or not value or '\x00' in value
            or '..' in value.split(os.sep)):
        raise IntentError(_INVALID)
    path = os.path.abspath(value)
    created = False
    if create:
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass          # pre-existing path: verify only, NEVER chmod it
        except OSError:
            raise IntentError(_INVALID) from None
    ok = False
    try:
        before = os.lstat(path)
        if (not stat.S_ISDIR(before.st_mode) or before.st_uid != os.geteuid()):
            raise IntentError(_INVALID)
        if not created and stat.S_IMODE(before.st_mode) != 0o700:
            raise IntentError(_INVALID)
        fd = os.open(path, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC)
        try:
            opened = os.fstat(fd)
            if (not stat.S_ISDIR(opened.st_mode) or opened.st_uid != os.geteuid()
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)):
                raise IntentError(_INVALID)
            if created:
                os.fchmod(fd, 0o700)   # own fresh directory: undo umask
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o700:
                raise IntentError(_INVALID)
            ok = True
            return fd, path
        except BaseException:
            os.close(fd)
            raise
    except IntentError:
        raise
    except OSError:
        raise IntentError(_INVALID) from None
    finally:
        if created and not ok:
            try:
                os.rmdir(path)
            except OSError:
                pass


def _open_intent(rootfd, key, *, fresh=False):
    try:
        before = os.stat(key, dir_fd=rootfd, follow_symlinks=False)
        if (not stat.S_ISDIR(before.st_mode) or before.st_uid != os.geteuid()
                or (not fresh and stat.S_IMODE(before.st_mode) != 0o700)):
            raise IntentError(_INVALID)
        fd = os.open(key, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                     dir_fd=rootfd)
        try:
            opened = os.fstat(fd)
            if (not stat.S_ISDIR(opened.st_mode) or opened.st_uid != os.geteuid()
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)):
                raise IntentError(_INVALID)
            if fresh:
                os.fchmod(fd, 0o700)   # own just-created directory: undo umask
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o700:
                raise IntentError(_INVALID)
        except BaseException:
            os.close(fd)
            raise
        return fd
    except IntentError:
        raise
    except OSError:
        raise IntentError(_INVALID) from None


def _write_exclusive(dirfd, name, data):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
                 | _CLOEXEC, 0o600, dir_fd=dirfd)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise IntentError(_INVALID)
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_checked(dirfd, name, budget):
    before = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 0 < before.st_size <= budget):
        raise IntentError(_INVALID)
    fd = os.open(name, os.O_RDONLY | _NOFOLLOW | _CLOEXEC, dir_fd=dirfd)
    try:
        opened = os.fstat(fd)
        if ((opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or stat.S_IMODE(opened.st_mode) != 0o600
                or stat.S_ISREG(opened.st_mode) is False):
            raise IntentError(_INVALID)
        parts, size = [], 0
        while True:
            piece = os.read(fd, min(_CHUNK, budget - size + 1))
            if not piece:
                break
            size += len(piece)
            if size > budget:
                raise IntentError(_INVALID)
            parts.append(piece)
        final = os.fstat(fd)
        after = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
        if (size != opened.st_size or final.st_size != opened.st_size
                or final.st_dev != opened.st_dev
                or final.st_ino != opened.st_ino
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)):
            raise IntentError(_INVALID)
        return b''.join(parts)
    finally:
        os.close(fd)


def _pairs(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise IntentError(_INVALID)
        document[key] = value
    return document


def _loads(raw):
    try:
        document = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs)
    except IntentError:
        raise
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
        raise IntentError(_INVALID) from None
    if type(document) is not dict or any(type(k) is not str for k in document):
        raise IntentError(_INVALID)
    return document


def _parse_manifest(raw, key):
    document = _loads(raw)
    keys = MANIFEST_KEYS | {'selected_tasks'} if 'selected_tasks' in document else MANIFEST_KEYS
    if (set(document) != keys
            or document['kind'] != INTENT_KIND
            or document['idempotency_key'] != key
            or document['ai_declaration'] not in client.AI_VALUES
            or type(document['source_sha256']) is not str
            or re.fullmatch('[0-9a-f]{64}', document['source_sha256']) is None):
        raise IntentError(_INVALID)
    if 'selected_tasks' in document:
        tasks = document['selected_tasks']
        try:
            canonical = client.canonical_selected_tasks(tasks)
        except client.ClientError:
            raise IntentError(_INVALID) from None
        if type(tasks) is not list or tasks != list(canonical):
            raise IntentError(_INVALID)
    return document


def _parse_receipt(raw):
    document = _loads(raw)
    if (set(document) != RECEIPT_KEYS
            or document['kind'] != RECEIPT_KIND):
        raise IntentError(_INVALID)
    _key(document['submission_id'])
    _utc(document['received_at'])
    return Receipt(document['submission_id'], document['received_at'])


def _discard(rootfd, intentfd, key):
    """Best-effort rollback of a just-created exclusive intent directory."""
    try:
        for name in (RECEIPT, MANIFEST, SOURCE):
            try:
                info = os.stat(name, dir_fd=intentfd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                    and stat.S_IMODE(info.st_mode) == 0o600):
                os.unlink(name, dir_fd=intentfd)
        os.rmdir(key, dir_fd=rootfd)
    except OSError:
        pass  # A leftover partial intent dir is refused by read_intent.


def _enforce_quota(rootfd, incoming):
    """Bounded cumulative-quota scan; the caller holds the exclusive root lock.

    Every directory entry counts toward MAX_INTENTS (foreign names included);
    bytes are summed from the regular files of key-shaped intent directories
    only. A same-privilege writer can hide bytes from the sum, matching the
    documented threat model; the count gate still bounds the layout.
    """
    try:
        names = os.listdir(rootfd)
    except OSError:
        raise IntentError(_INVALID) from None
    if len(names) >= MAX_INTENTS:
        raise IntentError(_INVALID)
    total = incoming
    for name in names:
        try:
            _key(name)
        except IntentError:
            continue
        try:
            info = os.stat(name, dir_fd=rootfd, follow_symlinks=False)
        except OSError:
            continue
        if not stat.S_ISDIR(info.st_mode):
            continue
        try:
            intentfd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW
                               | _CLOEXEC, dir_fd=rootfd)
        except OSError:
            continue
        try:
            for member in (SOURCE, MANIFEST, RECEIPT):
                try:
                    entry = os.stat(member, dir_fd=intentfd,
                                    follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISREG(entry.st_mode):
                    total += entry.st_size
        finally:
            os.close(intentfd)
        if total > MAX_TOTAL_INTENT_BYTES:
            raise IntentError(_INVALID)
    if total > MAX_TOTAL_INTENT_BYTES:
        raise IntentError(_INVALID)


def create_intent(root, *, payload, ai_declaration,
                  selected_tasks=client._SELECTION_OMITTED):
    """Persist one explicit new intent; omission is the legacy API branch."""
    _posix_primitives()
    tasks = None
    if selected_tasks is not client._SELECTION_OMITTED:
        try:
            tasks = client.canonical_selected_tasks(selected_tasks)
        except client.ClientError:
            raise IntentError(_INVALID) from None
    if ai_declaration not in client.AI_VALUES:
        raise IntentError(_INVALID)
    if (type(payload) is not bytes
            or not 0 < len(payload) <= client.MAX_ARCHIVE_BYTES):
        raise IntentError(_INVALID)
    rootfd, rootpath = _open_root(root, create=True)
    intentfd = None
    created = False
    try:
        try:
            fcntl.flock(rootfd, fcntl.LOCK_EX)
        except OSError:
            raise IntentError(_INVALID) from None
        _enforce_quota(rootfd, len(payload))
        key = client.new_submission_key()
        try:
            os.mkdir(key, 0o700, dir_fd=rootfd)
        except OSError:
            raise IntentError(_INVALID) from None
        created = True
        intentfd = _open_intent(rootfd, key, fresh=True)
        digest = hashlib.sha256(payload).hexdigest()
        _write_exclusive(intentfd, SOURCE, payload)
        try:
            package.validate_package(os.path.join(rootpath, key, SOURCE))
        except (package.PackageError, NotImplementedError, RuntimeError,
                EOFError, zlib.error):
            # Malformed archives must fold into the stable boundary error;
            # zipfile escapes these past BadZipFile (e.g. compression 99).
            raise IntentError(_INVALID) from None
        manifest = {'kind': INTENT_KIND, 'idempotency_key': key,
                    'source_sha256': digest, 'ai_declaration': ai_declaration}
        if tasks is not None:
            manifest['selected_tasks'] = list(tasks)
        _write_exclusive(intentfd, MANIFEST,
                         json.dumps(manifest, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8'))
        os.fsync(intentfd)
        return IntentRecord(key, digest, ai_declaration, payload, None, tasks)
    except BaseException:
        if created and intentfd is not None:
            _discard(rootfd, intentfd, key)
        raise
    finally:
        if intentfd is not None:
            os.close(intentfd)
        os.close(rootfd)


def read_intent(root, key):
    """Strict integrity re-read: same stored bytes, key and AI declaration."""
    _posix_primitives()
    _key(key)
    rootfd, _ = _open_root(root, create=False)
    intentfd = None
    try:
        intentfd = _open_intent(rootfd, key)
        try:
            names = set(os.listdir(intentfd))
        except OSError:
            raise IntentError(_INVALID) from None
        if (not {SOURCE, MANIFEST} <= names
                or not names <= {SOURCE, MANIFEST, RECEIPT}):
            raise IntentError(_INVALID)
        payload = _read_checked(intentfd, SOURCE, client.MAX_ARCHIVE_BYTES)
        digest = hashlib.sha256(payload).hexdigest()
        manifest = _parse_manifest(
            _read_checked(intentfd, MANIFEST, MAX_MANIFEST_BYTES), key)
        if manifest['source_sha256'] != digest:
            raise IntentError(_INVALID)
        receipt = None
        try:
            os.stat(RECEIPT, dir_fd=intentfd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            receipt = _parse_receipt(
                _read_checked(intentfd, RECEIPT, MAX_RECEIPT_BYTES))
        return IntentRecord(key, digest, manifest['ai_declaration'], payload,
                            receipt, None if 'selected_tasks' not in manifest
                            else tuple(manifest['selected_tasks']))
    except IntentError:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        raise IntentError(_INVALID) from None
    finally:
        if intentfd is not None:
            os.close(intentfd)
        os.close(rootfd)


def record_receipt(root, key, *, submission_id, received_at):
    """Write-once receipt; identical replay is safe, differences refused."""
    _posix_primitives()
    _key(key)
    _key(submission_id)
    _utc(received_at)
    read_intent(root, key)  # full integrity gate before any write
    rootfd, _ = _open_root(root, create=False)
    intentfd = None
    try:
        intentfd = _open_intent(rootfd, key)
        raw = json.dumps({'kind': RECEIPT_KIND, 'submission_id': submission_id,
                          'received_at': received_at}, sort_keys=True,
                         separators=(',', ':')).encode('utf-8')
        try:
            _write_exclusive(intentfd, RECEIPT, raw)
            os.fsync(intentfd)
        except FileExistsError:
            existing = _read_checked(intentfd, RECEIPT, MAX_RECEIPT_BYTES)
            if existing != raw:
                raise IntentError(_INVALID) from None
        return Receipt(submission_id, received_at)
    finally:
        if intentfd is not None:
            os.close(intentfd)
        os.close(rootfd)
