"""Windows local intent storage using standard-library SQLite transactions.

Only source ZIP bytes and closed intent/receipt metadata are stored. This is not
identity authentication. Paths reject observed symlinks/reparse points and
non-regular files, but do not claim POSIX owner/mode or full TOCTOU guarantees.
The backend can be exercised on another OS; that does not verify Windows locks.
"""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import zlib

from student_tools import intents as common

DATABASE = 'intents.sqlite3'
SCHEMA_VERSION = 1
_SCHEMA = ('CREATE TABLE intents (key TEXT PRIMARY KEY NOT NULL, '
           'payload BLOB NOT NULL, manifest BLOB NOT NULL, receipt BLOB) WITHOUT ROWID')
_REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)


def default_root():
    return Path.home() / '.ml-assignment-intents'


def _regular_path_info(info, *, directory=False):
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    return expected and not (getattr(info, 'st_file_attributes', 0) & _REPARSE)


def _check(path, *, directory=False):
    info = path.lstat()
    if not _regular_path_info(info, directory=directory):
        raise common.IntentError(common._INVALID)
    if not directory and info.st_nlink != 1:
        raise common.IntentError(common._INVALID)
    return info


def _root(root, *, create):
    path = Path(default_root() if root is None else root)
    if '..' in path.parts:
        raise common.IntentError(common._INVALID)
    path = Path(os.path.abspath(path))
    for parent in reversed(path.parents):
        _check(parent, directory=True)
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    _check(path, directory=True)
    _layout(path)
    return path


def _layout(root):
    for child in root.iterdir():
        if child.name not in (DATABASE, DATABASE + '-journal'):
            raise common.IntentError(common._INVALID)
        _check(child)


def _schema(conn):
    if conn.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
        raise common.IntentError(common._INVALID)
    rows = conn.execute('SELECT type,name,tbl_name,sql FROM sqlite_master').fetchall()
    if rows != [('table', 'intents', 'intents', _SCHEMA)]:
        raise common.IntentError(common._INVALID)
    if conn.execute('PRAGMA quick_check(1)').fetchone() != ('ok',):
        raise common.IntentError(common._INVALID)


def _quota(conn, incoming=0, *, extra_intent=0):
    count, total, bad = conn.execute(
        'SELECT count(*), coalesce(sum(length(payload)+length(manifest)+'
        'coalesce(length(receipt),0)),0), coalesce(sum('
        'typeof(key)!=\'text\' OR length(key)!=32 OR '
        'typeof(payload)!=\'blob\' OR length(payload)<1 OR length(payload)>? OR '
        'typeof(manifest)!=\'blob\' OR length(manifest)<1 OR length(manifest)>? OR '
        '(receipt IS NOT NULL AND (typeof(receipt)!=\'blob\' OR '
        'length(receipt)<1 OR length(receipt)>?))),0) FROM intents',
        (common.client.MAX_ARCHIVE_BYTES, common.MAX_MANIFEST_BYTES,
         common.MAX_RECEIPT_BYTES)).fetchone()
    if (bad or count + extra_intent > common.MAX_INTENTS
            or total + incoming > common.MAX_TOTAL_INTENT_BYTES):
        raise common.IntentError(common._INVALID)


@contextmanager
def _transaction(root, *, create=False):
    conn = None
    try:
        directory = _root(root, create=create)
        database = directory / DATABASE
        if create:
            try:
                descriptor = os.open(database, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        info = _check(database)
        # Bound the database before SQLite can parse arbitrary pages/schema.
        if info.st_size > common.MAX_TOTAL_INTENT_BYTES + 16 * 1024 * 1024:
            raise common.IntentError(common._INVALID)
        conn = sqlite3.connect(database.as_uri() + '?mode=rw', uri=True, timeout=30,
                               isolation_level=None)
        conn.execute('PRAGMA trusted_schema=OFF')
        conn.execute('PRAGMA synchronous=FULL')
        conn.execute('BEGIN IMMEDIATE')
        _layout(directory)
        _check(database)
        # Initialize only an empty version-zero database, under the same lock
        # as inserts. This also permits retry after a rolled-back first create.
        if (create and conn.execute('PRAGMA user_version').fetchone()[0] == 0
                and conn.execute('SELECT 1 FROM sqlite_master LIMIT 1').fetchone() is None):
            conn.execute(_SCHEMA)
            conn.execute('PRAGMA user_version=' + str(SCHEMA_VERSION))
        _schema(conn)
        _quota(conn)
        yield conn
        _layout(directory)
        _check(database)
        conn.commit()
    except common.IntentError:
        raise
    except (OSError, sqlite3.Error, ValueError, TypeError, OverflowError):
        raise common.IntentError(common._INVALID) from None
    finally:
        if conn is not None:
            try:
                if conn.in_transaction:
                    conn.rollback()
            finally:
                conn.close()


def _encode(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _insert(conn, key, payload, manifest):
    conn.execute('INSERT INTO intents(key,payload,manifest) VALUES(?,?,?)',
                 (key, payload, manifest))


def _materialize(conn, key):
    return conn.execute('SELECT payload,manifest,receipt FROM intents WHERE key=?', (key,)).fetchone()


def _read(conn, key):
    common._key(key)
    lengths = conn.execute(
        'SELECT length(payload),length(manifest),length(receipt),'
        'typeof(payload),typeof(manifest),typeof(receipt) FROM intents WHERE key=?', (key,)).fetchone()
    if (lengths is None or lengths[3:5] != ('blob', 'blob')
            or not 0 < lengths[0] <= common.client.MAX_ARCHIVE_BYTES
            or not 0 < lengths[1] <= common.MAX_MANIFEST_BYTES
            or (lengths[2] is not None and (lengths[5] != 'blob'
                or not 0 < lengths[2] <= common.MAX_RECEIPT_BYTES))):
        raise common.IntentError(common._INVALID)
    payload, raw, receipt_raw = _materialize(conn, key)
    manifest = common._parse_manifest(raw, key)
    digest = hashlib.sha256(payload).hexdigest()
    if manifest['source_sha256'] != digest:
        raise common.IntentError(common._INVALID)
    receipt = None if receipt_raw is None else common._parse_receipt(receipt_raw)
    tasks = None if 'selected_tasks' not in manifest else tuple(manifest['selected_tasks'])
    return common.IntentRecord(key, digest, manifest['ai_declaration'], payload, receipt, tasks)


def create_intent(root, *, payload, ai_declaration,
                  selected_tasks=common.client._SELECTION_OMITTED):
    tasks = None
    if selected_tasks is not common.client._SELECTION_OMITTED:
        try:
            tasks = common.client.canonical_selected_tasks(selected_tasks)
        except common.client.ClientError:
            raise common.IntentError(common._INVALID) from None
    if (type(ai_declaration) is not str or ai_declaration not in common.client.AI_VALUES
            or type(payload) is not bytes or not 0 < len(payload) <= common.client.MAX_ARCHIVE_BYTES):
        raise common.IntentError(common._INVALID)
    try:
        common.package.validate_package(io.BytesIO(payload))
    except (common.package.PackageError, NotImplementedError, RuntimeError, EOFError, zlib.error):
        raise common.IntentError(common._INVALID) from None
    key = common.client.new_submission_key()
    common._key(key)
    digest = hashlib.sha256(payload).hexdigest()
    document = {'kind': common.INTENT_KIND, 'idempotency_key': key,
                'source_sha256': digest, 'ai_declaration': ai_declaration}
    if tasks is not None:
        document['selected_tasks'] = list(tasks)
    manifest = _encode(document)
    if len(manifest) > common.MAX_MANIFEST_BYTES:
        raise common.IntentError(common._INVALID)
    with _transaction(root, create=True) as conn:
        _quota(conn, len(payload) + len(manifest), extra_intent=1)
        _insert(conn, key, payload, manifest)
    return common.IntentRecord(key, digest, ai_declaration, payload, None, tasks)


def read_intent(root, key):
    common._key(key)
    with _transaction(root) as conn:
        return _read(conn, key)


def record_receipt(root, key, *, submission_id, received_at):
    common._key(key)
    common._key(submission_id)
    common._utc(received_at)
    raw = _encode({'kind': common.RECEIPT_KIND, 'submission_id': submission_id,
                   'received_at': received_at})
    if len(raw) > common.MAX_RECEIPT_BYTES:
        raise common.IntentError(common._INVALID)
    with _transaction(root) as conn:
        record = _read(conn, key)
        if record.receipt is not None:
            existing = conn.execute('SELECT receipt FROM intents WHERE key=?', (key,)).fetchone()[0]
            if existing != raw:
                raise common.IntentError(common._INVALID)
        else:
            _quota(conn, len(raw))
            conn.execute('UPDATE intents SET receipt=? WHERE key=? AND receipt IS NULL', (raw, key))
    return common.Receipt(submission_id, received_at)
