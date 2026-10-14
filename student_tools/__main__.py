"""Single student tool entry. Production channel configuration is not supplied.

Trusted hosts may inject request; that callable is not proof of authorization.
An explicitly passed --endpoint builds the self-report request from the three
identity exports; those exports are metadata for attribution, never
authentication, and without --endpoint (and without an injected request)
online operations keep failing closed with zero network.
"""
import argparse
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import stat
import sys
import zlib

from . import client, identity, intents, package, self_report, selftest


class UsageError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError('INVALID_REQUEST')


def _tasks(value):
    try:
        return client.canonical_selected_tasks(value.split(','))
    except client.ClientError:
        raise argparse.ArgumentTypeError('choose four or five unique known tasks') from None


def _parser():
    parser = Parser(prog='python3 -B -m student_tools', description=(
        'Student tools: offline config, public selftest, package and saved submission '
        'intents. Online operations need an explicit self-report --endpoint or an '
        'injected host request; exports are attribution, not authentication. '
        'Selftest executes YOUR code locally, NOT in a sandbox.'))
    commands = parser.add_subparsers(dest='command')
    for name in ('config', 'selftest', 'package'):
        commands.add_parser(name, add_help=False)
    for name in ('prepare',):
        command = commands.add_parser(name)
        command.add_argument('--archive', required=True, help='existing source ZIP')
        command.add_argument('--ai', required=True, choices=sorted(client.AI_VALUES))
        command.add_argument('--tasks', required=True, type=_tasks,
                             help='comma-separated four or five tasks: lr,svm,gbdt,kmeans,hmm')
        command.add_argument('--store', default='.student-intents', help='private local intent directory')
    for name in ('intent',):
        command = commands.add_parser(name)
        command.add_argument('key', help='saved intent key (not an authentication token)')
        command.add_argument('--store', default='.student-intents')
    online = []
    command = commands.add_parser('submit')
    command.add_argument('--archive', required=True, help='existing source ZIP')
    command.add_argument('--ai', required=True, choices=sorted(client.AI_VALUES))
    command.add_argument('--tasks', required=True, type=_tasks,
                         help='comma-separated four or five tasks: lr,svm,gbdt,kmeans,hmm')
    command.add_argument('--store', default='.student-intents', help='private local intent directory')
    online.append(command)
    command = commands.add_parser('retry')
    command.add_argument('key', help='saved intent key (not an authentication token)')
    command.add_argument('--store', default='.student-intents')
    online.append(command)
    command = commands.add_parser('status')
    command.add_argument('submission_id')
    online.append(command)
    command = commands.add_parser('submissions')
    command.add_argument('--limit', type=int, default=20)
    command.add_argument('--offset', type=int, default=0)
    online.append(command)
    command = commands.add_parser('nickname')
    command.add_argument('nickname')
    online.append(command)
    # Explicit self-report endpoint, online subcommands only: the exports
    # stay metadata for attribution, never authentication.
    for command in online:
        command.add_argument('--endpoint', default=None,
                             help='explicit HTTPS origin for the self-report request')
    return parser


def _output(value):
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))


def _local_data(record):
    data = {'kind': 'local_intent', 'idempotency_key': record.idempotency_key,
            'source_sha256': record.source_sha256, 'ai_declaration': record.ai_declaration,
            'receipt': None if record.receipt is None else asdict(record.receipt)}
    if record.selected_tasks is not None:
        data['selected_tasks'] = list(record.selected_tasks)
    return data


def _read_archive(path):
    archive = Path(path).absolute()
    for part in (archive, *archive.parents):
        if stat.S_ISLNK(part.lstat().st_mode):
            raise package.PackageError('unsafe archive')
    fd = os.open(archive, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                 | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= client.MAX_ARCHIVE_BYTES:
            raise package.PackageError('archive budget')
        payload = stream.read(client.MAX_ARCHIVE_BYTES + 1)
        if not 0 < len(payload) <= client.MAX_ARCHIVE_BYTES:
            raise package.PackageError('archive budget')
        return payload


def _send(record, root, request):
    key = record.idempotency_key
    try:
        selection = {} if record.selected_tasks is None else {'selected_tasks': record.selected_tasks}
        data = client.submit(idempotency_key=key, payload=record.payload,
                             ai_declaration=record.ai_declaration, request=request, **selection)
    except client.ClientError as error:
        response = {'ok': False, 'error': {'code': error.code}, 'idempotency_key': key}
        if error.code in ('SERVICE_UNAVAILABLE', 'INVALID_RESPONSE'):
            response['outcome'] = 'unknown'
        _output(response)
        return 1
    except KeyboardInterrupt:
        _output({'ok': False, 'error': {'code': 'SERVICE_UNAVAILABLE'},
                 'idempotency_key': key, 'outcome': 'unknown'})
        return 1
    try:
        intents.record_receipt(root, key, submission_id=data['submission_id'],
                               received_at=data['received_at'])
    except (intents.IntentError, OSError, KeyboardInterrupt):
        # Server accepted; preserve its validated receipt even if local disk fails.
        _output({'ok': False, 'error': {'code': 'LOCAL_STATE_ERROR'},
                 'idempotency_key': key, 'data': data})
        return 1
    _output({'ok': True, 'idempotency_key': key, 'data': data})
    return 0


def main(argv=None, *, request=None):
    args = sys.argv[1:] if argv is None else list(argv)
    parser = _parser()
    if not args:
        parser.print_help()
        return 0
    if args[0] in ('config', 'selftest', 'package'):
        return {'config': identity.main, 'selftest': selftest.main,
                'package': package.main}[args[0]](args[1:])
    try:
        options = parser.parse_args(args)
    except UsageError:
        print('student tools: INVALID_REQUEST; use --help', file=sys.stderr)
        return 2
    except SystemExit as error:
        return error.code
    command = options.command
    endpoint = getattr(options, 'endpoint', None)
    if request is None and endpoint is not None:
        # Explicit self-report mode only: an injected request always wins,
        # and construction problems fail closed with a fixed, unechoed code.
        try:
            request = self_report.make_request(endpoint)
        except Exception:
            _output({'ok': False, 'error': {'code': 'SERVICE_NOT_CONFIGURED'}})
            return 1
    if command in ('prepare', 'submit'):
        try:
            payload = _read_archive(options.archive)
            package.validate_package(io.BytesIO(payload))
        except (OSError, ValueError, package.PackageError, RuntimeError, EOFError, zlib.error):
            _output({'ok': False, 'error': {'code': 'PACKAGE_REJECTED'}})
            return 1
        try:
            record = intents.create_intent(options.store, payload=payload,
                                           ai_declaration=options.ai,
                                           selected_tasks=options.tasks)
        except (intents.IntentError, OSError):
            _output({'ok': False, 'error': {'code': 'LOCAL_STATE_ERROR'}})
            return 1
        if command == 'prepare':
            _output({'ok': True, 'data': _local_data(record)})
            return 0
        return _send(record, options.store, request)
    if command in ('intent', 'retry'):
        try:
            record = intents.read_intent(options.store, options.key)
        except (intents.IntentError, OSError, ValueError):
            _output({'ok': False, 'error': {'code': 'LOCAL_STATE_ERROR'}})
            return 1
        if command == 'intent':
            _output({'ok': True, 'data': _local_data(record)})
            return 0
        return _send(record, options.store, request)
    try:
        if command == 'status':
            data = client.status(options.submission_id, request=request)
        elif command == 'submissions':
            data = client.list_submissions(limit=options.limit, offset=options.offset,
                                           request=request)
        else:
            data = client.set_nickname(options.nickname, request=request)
    except client.ClientError as error:
        _output({'ok': False, 'error': {'code': error.code}})
        return 1
    _output({'ok': True, 'data': data})
    return 0


if __name__ == '__main__':
    sys.exit(main())
