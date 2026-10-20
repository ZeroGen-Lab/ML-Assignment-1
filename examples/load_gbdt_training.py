"""Load the fixed student training CSV; no model fitting or prediction.

Example: python -I -B load_student_training04.py --csv /absolute/path/to/train.csv
Teacher evaluation files are not accepted. Output contains counts only.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

FEATURE_COUNT = 856
CLASS_LABELS = tuple(range(9))
EXPECTED_COUNTS = {label: 23 if label < 2 else 22 for label in CLASS_LABELS}
# Filled from the already fixed student manifest, never from a teacher CSV.
TRAIN_SHA256 = '214003a2c6c1af6f461867f9fca36ffda9c0fa183c4b8b13348a405031b83504'


def load_training(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('TRAIN_PATH')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 1024 * 1024:
            raise ValueError('TRAIN_FILE')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            raw = stream.read(1024 * 1024 + 1)
        after = os.fstat(fd)
        named = path.stat(follow_symlinks=False)
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_nlink')
        if any(getattr(before, key) != getattr(after, key) or
               getattr(before, key) != getattr(named, key) for key in keys):
            raise ValueError('TRAIN_DRIFT')
    finally:
        os.close(fd)
    if hashlib.sha256(raw).hexdigest() != TRAIN_SHA256:
        raise ValueError('TRAIN_SHA256')
    reader = csv.reader(io.StringIO(raw.decode('ascii')))
    if next(reader, None) != ['label'] + [f'x{i:03d}' for i in range(FEATURE_COUNT)]:
        raise ValueError('TRAIN_HEADER')
    features, labels = [], []
    counts = {label: 0 for label in CLASS_LABELS}
    for row in reader:
        if len(row) != FEATURE_COUNT + 1 or any(re.fullmatch(r'[0-9]+', cell) is None for cell in row):
            raise ValueError('TRAIN_ROW')
        values = list(map(int, row))
        if values[0] not in CLASS_LABELS or any(x < 0 or x > 4 for x in values[1:]):
            raise ValueError('TRAIN_VALUE')
        labels.append(values[0])
        features.append(values[1:])
        counts[values[0]] += 1
    if len(labels) != 200 or counts != EXPECTED_COUNTS:
        raise ValueError('TRAIN_COUNTS')
    return features, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()
    x, y = load_training(args.csv)
    print(json.dumps({'records': len(y), 'feature_count': len(x[0]),
                      'class_labels': list(CLASS_LABELS), 'training': 'NOT_RUN',
                      'predict': 'NOT_RUN', 'metrics': 'NOT_RUN'}, sort_keys=True))
