"""Public advisor GBDT data/protocol checks; never compiles or executes student code.

The smoke query is the public training set itself, not teacher evaluation data.
A successful check proves only count/integer/domain compatibility, not ML quality.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import types


LOADER_SHA256 = 'a75013365275b6d458ebbf4e0f28e375cd697ace5958f7a382f7398fb09125b0'
PROTOCOL_VERSION = 'advisor-public-gbdt9-count-vector-v1'
TOKEN = re.compile(r'(?:0|[1-9][0-9]*)\Z', re.ASCII)


def read_fixed(path, cap):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('ABSOLUTE_PATH_REQUIRED')
    for parent in reversed(path.parents):
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError('UNSAFE_PARENT')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > cap:
            raise ValueError('UNSAFE_FILE')
        raw = source.read(cap + 1)
        after = os.fstat(source.fileno())
        named = path.stat(follow_symlinks=False)
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len(raw) > cap or identity(before) != identity(after) or identity(after) != identity(named):
        raise ValueError('FILE_CHANGED')
    return raw


def load_training(csv_path, loader_path):
    raw = read_fixed(loader_path, 16384)
    if hashlib.sha256(raw).hexdigest() != LOADER_SHA256:
        raise ValueError('UNREVIEWED_TRAINING_LOADER')
    module = types.ModuleType('fixed_advisor_training_loader')
    module.__file__ = str(loader_path)
    exec(compile(raw, str(loader_path), 'exec'), module.__dict__)
    return module.load_training(csv_path)


def encode_training_smoke(features, labels, seed=0):
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError('SEED')
    if len(features) != 200 or len(labels) != 200:
        raise ValueError('TRAINING_COUNT')
    if any(type(y) is not int or not 0 <= y < 9 for y in labels):
        raise ValueError('TRAINING_LABEL')
    if any(len(row) != 856 or any(type(v) is not int or not 0 <= v <= 4 for v in row)
           for row in features):
        raise ValueError('TRAINING_FEATURES')
    # Public count-vector protocol: this is not formal FAI2/FAO2.
    lines = [f'gbdt {seed} 200 856 200']
    rows = [' '.join(map(str, row)) for row in features]
    lines.extend(rows)
    lines.append(' '.join(map(str, labels)))
    lines.extend(rows)
    return ('\n'.join(lines) + '\n').encode('ascii')


def parse_predictions(raw, expected=200):
    if type(raw) is not bytes or len(raw) > 16384:
        raise ValueError('OUTPUT_BYTES')
    if type(expected) is not int or expected != 200:
        raise ValueError('PREDICTION_COUNT')
    try:
        tokens = raw.decode('ascii').split()
    except UnicodeDecodeError:
        raise ValueError('OUTPUT_ASCII') from None
    if len(tokens) != expected + 1 or tokens[0] != str(expected):
        raise ValueError('OUTPUT_COUNT')
    if any(not TOKEN.fullmatch(t) or len(t) != 1 or not 0 <= int(t) < 9
           for t in tokens[1:]):
        raise ValueError('OUTPUT_INTEGER_CLASS')
    return [int(t) for t in tokens[1:]]


def write_new(path, raw):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('ABSOLUTE_PATH_REQUIRED')
    for parent in reversed(path.parents):
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError('UNSAFE_PARENT')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    if read_fixed(path, len(raw)) != raw:
        raise ValueError('WRITE_MISMATCH')
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='write public training-only smoke stdin')
    prepare.add_argument('--csv', type=Path, required=True)
    prepare.add_argument('--loader', type=Path,
                         default=Path(__file__).absolute().parent.parent / 'examples/load_gbdt_training.py')
    prepare.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    prepare.add_argument('--out', type=Path, required=True)
    check = commands.add_parser('check', help='check a previously obtained output file')
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        features, labels = load_training(args.csv, args.loader)
        raw = encode_training_smoke(features, labels, args.seed)
        write_new(args.out, raw)
        status = {'status': 'TRAINING_SMOKE_INPUT_ONLY', 'stdin_bytes': len(raw),
                  'stdin_sha256': hashlib.sha256(raw).hexdigest()}
    else:
        predictions = parse_predictions(read_fixed(args.output, 16384))
        status = {'status': 'OUTPUT_SHAPE_DOMAIN_ONLY', 'prediction_count': len(predictions)}
    status.update(protocol_version=PROTOCOL_VERSION, compile='NOT_RUN', training='NOT_RUN',
                  metrics='NOT_RUN', reference_score=None)
    print(json.dumps(status, sort_keys=True))


if __name__ == '__main__':
    main()
