"""Public local self-test for the five ML-Assignment-1 APIs.

CLI (see docs/student-guide.md):

    python3 -B -m student_tools.selftest --project starter [--compiler clang++]

This compiles the course-provided fixed driver (``<project>/tools/driver.cpp``)
together with the student sources under ``src/`` and headers under
``include/``, then runs five small PUBLIC fixtures through the driver and
checks the outputs (finite values, label domains, exact lengths, no trailing
output) plus simple public quality checks.  An unfilled template must fail
honestly - a successful compile alone never counts as a working algorithm.

Runs on YOUR OWN machine and executes YOUR C++ code: this is a convenience
self-test, not a sandbox and not official grading.  The official grading
environment uses GCC 12 with C++17 and -O2; local results with other
compilers (e.g. Apple clang) are compatibility checks only.

Exit codes: 0 all five tasks pass; 1 at least one task failed (compile
error, runtime error, timeout, invalid output, or failed quality check);
3 environment problem (missing compiler/project/driver).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASKS = ('lr', 'svm', 'gbdt', 'kmeans', 'hmm')
COMPILER_CANDIDATES = ('g++', 'clang++')
MAX_OUTPUT_BYTES = 1_048_576          # cap on captured program output
STDERR_SNIPPET = 400                  # how much of an error stream to show
OFFICIAL_TOOLCHAIN = 'GCC 12 + C++17 + -O2'


class EnvironmentError_(Exception):
    """The local environment cannot run the self-test at all."""


class InvalidOutput(Exception):
    """The binary answered outside the public protocol."""


# ---------------------------------------------------------------- fixtures

def _numbers(values):
    return ' '.join(repr(float(v)) for v in values)


def _ints(values):
    return ' '.join(str(int(v)) for v in values)


def classification_fixture():
    """Two cleanly separated 2-D blobs; deterministic, no RNG."""
    def points(center, count):
        return [[center[0] + 0.35 * (i % 4) - 0.5, center[1] + 0.4 * (i // 4) - 0.4]
                for i in range(count)]
    train_x = points((-2.6, -2.4), 12) + points((2.6, 2.4), 12)
    train_y = [0] * 12 + [1] * 12
    test_x = points((-2.4, -2.6), 4) + points((2.4, 2.6), 4)
    truth = [0] * 4 + [1] * 4
    payload = '\n'.join([TASK_TOKEN_PLACEHOLDER, '7',
                         str(len(train_x)), '2', str(len(test_x)),
                         *[_numbers(row) for row in train_x],
                         _ints(train_y),
                         *[_numbers(row) for row in test_x]]) + '\n'
    return payload, {'truth': truth}


def regression_fixture():
    """A step in feature 0 plus a mild linear term in feature 1."""
    def rows(count, shift):
        return [[4.0 * i / (count - 1) + shift, -1.0 + 2.0 * i / (count - 1)] for i in range(count)]
    def target(row):
        return (1.5 if row[0] > 2.0 else -1.0) + 0.2 * row[1]
    train_x, test_x = rows(32, 0.0), rows(16, 0.12)
    train_y = [target(row) for row in train_x]
    truth = [target(row) for row in test_x]
    payload = '\n'.join(['gbdt', '11',
                         str(len(train_x)), '2', str(len(test_x)),
                         *[_numbers(row) for row in train_x],
                         _numbers(train_y),
                         *[_numbers(row) for row in test_x]]) + '\n'
    return payload, {'truth': truth, 'y_train': train_y}


def clustering_fixture():
    """Four separated 4-D training blobs and distinct prediction rows; no RNG."""
    train, test, truth = [], [], []
    for label in range(4):
        center = -6.0 + 4.0 * label
        for i in range(9):
            train.append([center + 0.1 * (i % 3 - 1), 0.1 * (i // 3 - 1),
                          0.05 * (i % 2), 0.02 * i])
        for i in range(3):
            test.append([center + 0.03 * (i - 1), 0.04 * i, 0.02, 0.01])
            truth.append(label)
    payload = '\n'.join(['kmeans', '13', str(len(train)), '4', str(len(test)), '4',
                         *[_numbers(row) for row in train],
                         *[_numbers(row) for row in test]]) + '\n'
    return payload, {'truth': truth, 'k': 4}


def hmm_fixture():
    """Supervised 3-state/4-symbol micro-fixture; not the official effect data."""
    train = [[0, 1, 2, 3, 0, 1], [2, 1, 0, 3, 2, 1], [3, 2, 1, 0]]
    labels = [[symbol % 3 for symbol in seq] for seq in train]
    observations = [[0, 2, 1, 3], [2, 2, 0], [1], []]
    lines = ['hmm', '0', '3', '4', str(len(train)), str(len(observations))]
    for group in (train, labels, observations):
        lines += [' '.join([str(len(seq)), *map(str, seq)]) for seq in group]
    expected = [[symbol % 3 for symbol in seq] for seq in observations]
    return '\n'.join(lines) + '\n', {'observations': observations, 'expected': expected,
                                     'states': 3}


TASK_TOKEN_PLACEHOLDER = '__task__'
FIXTURES = {'lr': classification_fixture, 'svm': classification_fixture,
            'gbdt': regression_fixture, 'kmeans': clustering_fixture, 'hmm': hmm_fixture}


# ----------------------------------------------------------------- oracles

def viterbi_path(a, b, pi, observations):
    """Public oracle: joint-optimal path, log domain, ties -> smallest predecessor/state."""
    if not observations:
        return []
    states = range(len(pi))
    neg = -math.inf

    def logp(p):
        return math.log(p) if p > 0 else neg

    delta = [logp(pi[i]) + logp(b[i][observations[0]]) for i in states]
    back = []
    for obs in observations[1:]:
        new_delta, row = [], []
        for j in states:
            best_i, best_score = 0, neg
            for i in states:
                score = delta[i] + logp(a[i][j])
                if score > best_score:
                    best_i, best_score = i, score
            row.append(best_i)
            new_delta.append(best_score + logp(b[j][obs]))
        back.append(row)
        delta = new_delta
    last = min((s for s in states if delta[s] == max(delta)), default=0)
    path = [last]
    for row in reversed(back):
        path.append(row[path[-1]])
    return list(reversed(path))


def _pair_counts(truth, predictions):
    counts = {}
    for pair in zip(truth, predictions):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def ari(truth, predictions):
    """Small public ARI (clustering labels are permutation invariant)."""
    n = len(truth)
    joint = _pair_counts(truth, predictions)
    rows, cols = {}, {}
    for (t, p), v in joint.items():
        rows[t] = rows.get(t, 0) + v
        cols[p] = cols.get(p, 0) + v
    choose2 = lambda v: v * (v - 1) / 2
    total = choose2(n)
    index = sum(choose2(v) for v in joint.values())
    expected = sum(choose2(v) for v in rows.values()) * sum(choose2(v) for v in cols.values()) / total
    denominator = (sum(choose2(v) for v in rows.values()) +
                   sum(choose2(v) for v in cols.values())) / 2 - expected
    return 1.0 if abs(denominator) < 1e-15 else (index - expected) / denominator


# ------------------------------------------------------------- compilation

def find_compiler(explicit=None):
    if explicit:
        resolved = shutil.which(explicit)
        if not resolved:
            raise EnvironmentError_(f'compiler not found: {explicit}')
        return resolved
    for candidate in COMPILER_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise EnvironmentError_('no C++17 compiler found (tried: %s); pass --compiler PATH'
                            % ', '.join(COMPILER_CANDIDATES))


def compile_project(compiler, project, workdir, timeout):
    """Compile the fixed driver plus the student sources; return the executable path."""
    project = Path(project)
    include, src = project / 'include', project / 'src'
    driver = project / 'tools' / 'driver.cpp'
    for required in (include, src, driver):
        if not Path(required).exists():
            raise EnvironmentError_(f'missing {required}')
    sources = sorted(str(p) for p in src.rglob('*.cpp'))
    if not sources:
        raise EnvironmentError_(f'no .cpp sources under {src}')
    include_flags = ['-I', str(include)]
    for extra in sorted(p for p in include.rglob('*') if p.is_dir()):
        include_flags += ['-I', str(extra)]
    executable = Path(workdir) / 'selftest-driver'
    command = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', *include_flags,
               str(driver), *sources, '-o', str(executable)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f'compilation exceeded {timeout:.0f}s'
    except OSError as error:
        return None, f'could not run compiler: {error}'
    if done.returncode != 0:
        snippet = (done.stderr or done.stdout or '').strip()
        return None, 'compiler rejected the sources (first 1200 chars):\n' + snippet[:1200]
    return executable, None


# ----------------------------------------------------------------- running

def _kill_process_group(process):
    try:
        if platform.system() == 'Windows':
            process.kill()
        else:
            import signal
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _read_snippet(path):
    """Read at most STDERR_SNIPPET bytes of a redirected stream file.

    A chatty binary can write far more to stderr than the snippet we keep;
    load only the leading bytes instead of the whole file. This bounds the
    in-memory read only -- the redirected file on disk is NOT bounded here.
    """
    with open(path, 'rb') as stream:
        return stream.read(STDERR_SNIPPET)


def run_binary(executable, payload, timeout, workdir):
    """Run one task; output goes to files so a chatty binary cannot exhaust memory."""
    with tempfile.TemporaryDirectory(prefix='selftest-run-', dir=workdir) as rundir:
        out_path, err_path = Path(rundir) / 'out', Path(rundir) / 'err'
        kwargs = {}
        if platform.system() == 'Windows':
            kwargs['creationflags'] = 0x00000200       # CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True
        with open(out_path, 'wb') as out, open(err_path, 'wb') as err:
            process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                       stdout=out, stderr=err, **kwargs)
            try:
                process.communicate(payload.encode('utf-8'), timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                return 'timeout', '', ''
        if process.returncode != 0:
            snippet = _read_snippet(err_path).decode('utf-8', 'replace').strip()
            return 'runtime_error', '', snippet
        if out_path.stat().st_size > MAX_OUTPUT_BYTES:
            return 'invalid_output', '', 'output exceeds %d bytes' % MAX_OUTPUT_BYTES
        text = out_path.read_bytes().decode('utf-8', 'replace')
        return 'ok', text, _read_snippet(err_path).decode('utf-8', 'replace')


# ---------------------------------------------------------------- checking

_INT = re.compile(r'^[+-]?\d+$')


def _parse_vector(tokens, count, converter, domain, label):
    values = []
    for _ in range(count):
        token = next(tokens, None)
        if token is None:
            raise InvalidOutput(f'{label}: fewer values than announced')
        if converter is int:
            if not _INT.match(token):
                raise InvalidOutput(f'{label}: non-integer value {token!r}')
            value = int(token)
        else:
            try:
                value = float(token)
            except ValueError:
                raise InvalidOutput(f'{label}: non-numeric value {token!r}') from None
            if not math.isfinite(value):
                raise InvalidOutput(f'{label}: non-finite value {token!r}')
        if domain is not None and value not in domain:
            raise InvalidOutput(f'{label}: value {value!r} outside {domain}')
        values.append(value)
    return values


def parse_output(task, text, spec):
    tokens = iter(text.split())
    try:
        count = int(next(tokens))
    except (StopIteration, ValueError):
        raise InvalidOutput('missing output count') from None
    if task == 'hmm':
        if count != len(spec['observations']):
            raise InvalidOutput(f'announced {count} sequences, expected {len(spec["observations"])}')
        paths = []
        for index, sequence in enumerate(spec['observations']):
            head = next(tokens, None)
            if head is None:
                raise InvalidOutput(f'sequence {index}: missing length')
            if not _INT.match(head):
                raise InvalidOutput(f'sequence {index}: bad length token {head!r}')
            length = int(head)
            if length != len(sequence):
                raise InvalidOutput(f'sequence {index}: length {length}, expected {len(sequence)}')
            paths.append(_parse_vector(tokens, length, int, range(spec['states']),
                                       f'sequence {index}'))
        prediction = paths
    else:
        expected = spec['prediction_count'] if 'prediction_count' in spec else len(spec['truth'])
        if count != expected:
            raise InvalidOutput(f'announced {count} values, expected {expected}')
        if task == 'kmeans':
            domain = range(spec['k'])
        elif task in ('lr', 'svm'):
            domain = range(2)
        else:
            domain = None
        prediction = _parse_vector(tokens, count, float if task == 'gbdt' else int,
                                   domain, task)
    if next(tokens, None) is not None:
        raise InvalidOutput('trailing output after announced values')
    return prediction


def quality_check(task, prediction, spec):
    """Simple PUBLIC sanity thresholds; passing them is not a course grade."""
    if task in ('lr', 'svm'):
        accuracy = sum(t == p for t, p in zip(spec['truth'], prediction)) / len(spec['truth'])
        return accuracy >= 0.9, f'accuracy={accuracy:.3f}'
    if task == 'gbdt':
        mean = sum(spec['y_train']) / len(spec['y_train'])
        baseline = math.sqrt(sum((t - mean) ** 2 for t in spec['truth']) / len(spec['truth']))
        rmse = math.sqrt(sum((t - p) ** 2 for t, p in zip(spec['truth'], prediction))
                         / len(spec['truth']))
        return rmse <= 0.75 * baseline, f'rmse={rmse:.3f} baseline={baseline:.3f}'
    if task == 'kmeans':
        score = ari(spec['truth'], prediction)
        return score > 0.8, f'ari={score:.3f}'
    exact = sum(path == want for path, want in zip(prediction, spec['expected']))
    return exact == len(spec['expected']), f'exact_sequences={exact}/{len(spec["expected"])}'


def run_task(executable, task, timeout, workdir, public_validation=None):
    payload, spec = public_validation if public_validation is not None else FIXTURES[task]()
    payload = payload.replace(TASK_TOKEN_PLACEHOLDER, task, 1)
    category, text, detail = run_binary(executable, payload, timeout, workdir)
    if category != 'ok':
        return category, detail
    try:
        prediction = parse_output(task, text, spec)
    except InvalidOutput as error:
        return 'invalid_output', str(error)
    passed, detail = (public_validation_metrics(task, prediction, spec)
                      if public_validation is not None else quality_check(task, prediction, spec))
    return ('ok', detail) if passed else ('quality_check_failed', detail)


# ------------------------------------------------------ public validation

def _public_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _public_input_spec(task, payload, *, data_version='course-single-20261013-v1'):
    if data_version not in ('course-single-20261013-v1', 'course-uci-20261014-v1'):
        raise ValueError('unknown public data version')
    uci = data_version == 'course-uci-20261014-v1'
    tokens = iter(payload.split())
    if next(tokens, None) != task:
        raise ValueError('input task mismatch')
    def integer(expected=None, domain=None):
        token = next(tokens)
        if not re.fullmatch(r'0|[1-9][0-9]*', token):
            raise ValueError('invalid input integer')
        value = int(token)
        if (expected is not None and value != expected) or (domain is not None and value not in domain):
            raise ValueError('input dimension or domain mismatch')
        return value
    def numbers(count):
        for _ in range(count):
            if not math.isfinite(float(next(tokens))):
                raise ValueError('non-finite input')
    integer(0)
    if task == 'hmm':
        integer(3); integer(4); integer(8); integer(2)
        sequences = []
        for index in range(18):
            integer(20)
            sequences.append([integer(domain=range(3 if 8 <= index < 16 else 4))
                              for _ in range(20)])
        spec = {'states': 3, 'observations': sequences[16:]}
    else:
        dimension = ({'lr': 30, 'svm': 57, 'gbdt': 4, 'kmeans': 7} if uci else
                     {'lr': 8, 'svm': 8, 'gbdt': 6, 'kmeans': 4})[task]
        train_count, test_count, clusters = (120, 30, 3) if uci else (160, 40, 4)
        integer(train_count); integer(dimension); integer(test_count)
        if task == 'kmeans':
            integer(clusters)
        numbers(train_count * dimension)
        if task in ('lr', 'svm'):
            for _ in range(train_count):
                integer(domain=range(2))
        elif task == 'gbdt':
            numbers(train_count)
        numbers(test_count * dimension)
        spec = {'prediction_count': test_count}
        if task == 'kmeans':
            spec['k'] = clusters
    if next(tokens, None) is not None:
        raise ValueError('trailing input tokens')
    return spec


def load_public_validation(directory, task):
    """Read A's public text/answer pair; hashes detect drift, not authenticity."""
    try:
        if task not in TASKS:
            raise ValueError('unknown task')
        directory = Path(directory)
        def read(name, limit):
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
                raise ValueError('missing, linked or oversized public file')
            with path.open('rb') as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise ValueError('oversized public file')
            return raw
        manifest = _public_json(read('manifest.json', 65536))
        legacy_keys = {'status', 'input_manifest', 'input_manifest_sha256', 'data_version',
                       'dataset_id', 'files', 'driver_executed', 'student_executed', 'gates'}
        public_keys = {'schema', 'data_version', 'dataset_id', 'files'}
        if (not isinstance(manifest, dict)
                or not ((set(manifest) == legacy_keys
                         and manifest['status'] == 'PUBLIC_VALIDATION_CANDIDATE_NOT_PUBLISHED')
                        or (set(manifest) == public_keys
                            and manifest['schema'] == 'course-public-validation-v1'))
                or manifest['data_version'] not in ('course-single-20261013-v1', 'course-uci-20261014-v1')
                or not isinstance(manifest['dataset_id'], str)
                or not re.fullmatch('[0-9a-f]{64}', manifest['dataset_id'])
                or not isinstance(manifest['files'], list) or len(manifest['files']) != 10):
            raise ValueError('invalid public manifest')
        uci = manifest['data_version'] == 'course-uci-20261014-v1'
        if uci and (set(manifest) != public_keys or manifest['dataset_id'] !=
                    'db695906b8e038e25f82cec2e4eaa7e5224eaa8121eab0b7f569184a14457529'):
            raise ValueError('invalid UCI public identity or schema')
        expected = {t + suffix: t for t in TASKS for suffix in ('.stdin', '.validation.json')}
        files = {}
        for entry in manifest['files']:
            if not isinstance(entry, dict) or set(entry) != {'name', 'task', 'bytes', 'sha256'}:
                raise ValueError('invalid file entry')
            name = entry['name']
            if (not isinstance(name, str) or name not in expected or name in files
                    or entry['task'] != expected[name] or type(entry['bytes']) is not int
                    or not 0 < entry['bytes'] <= MAX_OUTPUT_BYTES
                    or not isinstance(entry['sha256'], str)
                    or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])):
                raise ValueError('invalid file identity')
            raw = read(name, MAX_OUTPUT_BYTES)
            if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('public file identity mismatch: ' + name)
            files[name] = raw
        metadata = _public_json(files[task + '.validation.json'])
        modes = {'lr': 'classification_accuracy_f1', 'svm': 'classification_accuracy_f1',
                 'gbdt': 'regression_mae_rmse', 'kmeans': 'unlabelled_diagnostics_not_ari_nmi',
                 'hmm': 'hmm_state_accuracy'}
        if (not isinstance(metadata, dict) or set(metadata) != {
                'data_version', 'dataset_id', 'evaluation_mode', 'task',
                'validation_answers', 'validation_ids'} or metadata['task'] != task
                or metadata['evaluation_mode'] != modes[task]
                or any(metadata[key] != manifest[key] for key in ('data_version', 'dataset_id'))):
            raise ValueError('invalid validation metadata')
        ids = metadata['validation_ids']
        count = 2 if task == 'hmm' else (30 if uci else 40)
        family = 'classification' if task in ('lr', 'svm') else task
        upper_id = 90 if task == 'hmm' else 1800
        if uci:
            family, upper_id = {'lr': ('wdbc', 569), 'svm': ('spambase', 4204),
                                'gbdt': ('ccpp', 9527), 'kmeans': ('seeds', 210),
                                'hmm': ('hmm', 90)}[task]
        if (not isinstance(ids, list) or len(ids) != count
                or any(not isinstance(value, str) or not re.fullmatch(family + r':(0|[1-9][0-9]*)', value)
                       or int(value.split(':')[1]) >= upper_id for value in ids)
                or len(set(ids)) != count):
            raise ValueError('invalid validation IDs')
        answers = metadata['validation_answers']
        if task == 'kmeans':
            valid = answers is None
        else:
            valid = isinstance(answers, list) and len(answers) == count
            if valid and task == 'hmm':
                valid = all(isinstance(seq, list) and len(seq) == 20
                            and all(type(x) is int and x in range(3) for x in seq) for seq in answers)
            elif valid and task in ('lr', 'svm'):
                valid = all(type(x) is int and x in range(2) for x in answers)
            elif valid:
                valid = all(type(x) in (int, float) and math.isfinite(x) for x in answers)
        if not valid:
            raise ValueError('invalid validation answers')
        payload = files[task + '.stdin'].decode('utf-8')
        if not payload.endswith('\n'):
            raise ValueError('public input lacks final newline')
        spec = _public_input_spec(task, payload, data_version=manifest['data_version'])
        spec.update(metadata)
        spec['truth'] = answers
        return payload, spec
    except (OSError, ValueError, TypeError, KeyError, StopIteration, OverflowError) as error:
        raise EnvironmentError_('invalid public validation data: ' + str(error)) from None


def public_validation_metrics(task, prediction, spec):
    """Diagnostics only: legal low-quality outputs are not threshold failures."""
    truth = spec['truth']
    if task in ('lr', 'svm'):
        accuracy = sum(t == p for t, p in zip(truth, prediction)) / len(truth)
        scores = []
        for label in range(2):
            tp = sum(t == p == label for t, p in zip(truth, prediction))
            denominator = sum(t == label for t in truth) + sum(p == label for p in prediction)
            scores.append(2 * tp / denominator if denominator else 0.0)
        detail = f'accuracy={accuracy:.3f} macro_f1={sum(scores) / 2:.3f}'
    elif task == 'gbdt':
        errors = [abs(t - p) for t, p in zip(truth, prediction)]
        mae = sum(error / len(errors) for error in errors)
        rmse = math.hypot(*errors) / math.sqrt(len(errors))
        detail = f'mae={mae:.3f} rmse={rmse:.3f}'
    elif task == 'hmm':
        total = sum(map(len, truth))
        correct = sum(t == p for ts, ps in zip(truth, prediction) for t, p in zip(ts, ps))
        exact = sum(ts == ps for ts, ps in zip(truth, prediction))
        detail = f'state_accuracy={correct / total:.3f} exact_sequences={exact}/{len(truth)}'
    else:
        counts = [prediction.count(label) for label in range(spec['k'])]
        detail = f'unlabelled occupied_clusters={sum(n > 0 for n in counts)}/{spec["k"]} counts={counts}'
    return True, 'local_diagnostic ' + detail + ' (not official grading)'


# --------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python3 -m student_tools.selftest',
        description=('Compile and run five small PUBLIC API checks against the fixed course '
                     'driver. Local convenience self-test on your own machine: NOT a sandbox '
                     'and NOT official grading. The official environment uses %s; '
                     'results with other compilers are compatibility checks only.'
                     % OFFICIAL_TOOLCHAIN))
    parser.add_argument('--project', default='starter',
                        help='project directory with include/, src/ and tools/driver.cpp')
    parser.add_argument('--compiler', help='compiler command or path (default: try %s)'
                        % ' then '.join(COMPILER_CANDIDATES))
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='per-task run timeout in seconds (default: 10)')
    parser.add_argument('--compile-timeout', type=float, default=120.0,
                        help='compilation timeout in seconds (default: 120)')
    parser.add_argument('--tasks', default=','.join(TASKS),
                        help='comma-separated four or five distinct tasks (default: all five)')
    parser.add_argument('--public-data', help='directory containing public validation manifest, stdin and answers; diagnostics only')
    args = parser.parse_args(argv)
    requested = args.tasks.split(',')
    if (len(requested) not in (4, 5) or len(set(requested)) != len(requested)
            or any(task not in TASKS for task in requested)):
        parser.error('--tasks requires four or five distinct names from ' + ','.join(TASKS))
    selected_tasks = tuple(task for task in TASKS if task in requested)

    print('ML-Assignment-1 public self-test')
    print('note: runs on your own machine and executes your C++ code; not a sandbox')
    print(f'note: official grading uses {OFFICIAL_TOOLCHAIN}; '
          'local compiler results are compatibility checks only')
    try:
        public = ({task: load_public_validation(args.public_data, task) for task in selected_tasks}
                  if args.public_data is not None else None)
        compiler = find_compiler(args.compiler)
    except EnvironmentError_ as error:
        print(f'selftest: environment error: {error}', file=sys.stderr)
        return 3
    print(f'compiler: {compiler}')
    print(f'project: {args.project}')

    with tempfile.TemporaryDirectory(prefix='ml-selftest-') as workdir:
        try:
            executable, error = compile_project(compiler, args.project, workdir,
                                                args.compile_timeout)
        except EnvironmentError_ as error:
            print(f'selftest: environment error: {error}', file=sys.stderr)
            return 3
        if executable is None:
            print('compile: FAIL compile_error')
            print((error or '').strip())
            print(f'result: 0/{len(selected_tasks)} passed')
            print('SELFTEST FAILED (compile_error)')
            return 1
        print('compile: ok')
        failures = 0
        for task in selected_tasks:
            started = time.monotonic()
            category, detail = (run_task(executable, task, args.timeout, workdir, public[task])
                                if public is not None else run_task(executable, task, args.timeout, workdir))
            elapsed = time.monotonic() - started
            status = 'ok' if category == 'ok' else f'FAIL {category}'
            line = f'task {task:<7}: {status:<22} ({elapsed:5.1f}s)'
            if detail:
                line += ' ' + detail.replace('\n', ' | ')[:300]
            print(line)
            if category != 'ok':
                failures += 1
    print(f'result: {len(selected_tasks) - failures}/{len(selected_tasks)} passed')
    print('SELFTEST PASSED' if failures == 0 else f'SELFTEST FAILED ({failures} task(s) failed)')
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
