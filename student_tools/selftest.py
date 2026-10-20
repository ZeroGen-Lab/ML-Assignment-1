"""Local compilation and output checks for the course C++ interfaces."""

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


PUBLIC_SELFTEST_VERSION = 'advisor-public-selftest13-v1'
POLICY_VERSION = 'trial-quality25-advisor-20261020-v1'
PUBLIC_CHECK_SHA256 = '494971615fc5d0700a225f60a56560840ca5f22bdaa3c8a36578aba40da8bf2e'


def _gbdt_check():
    """Load only the fixed companion, including under Python isolated mode."""
    import types
    path = Path(__file__).absolute().with_name('advisor_gbdt_public_check.py')
    try:
        with path.open('rb') as handle:
            raw = handle.read(32769)
        if len(raw) > 32768 or hashlib.sha256(raw).hexdigest() != PUBLIC_CHECK_SHA256:
            raise ValueError('UNREVIEWED_PUBLIC_CHECK')
        module = types.ModuleType('fixed_advisor_public_check13')
        module.__file__ = str(path)
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        return module
    except (OSError, ValueError) as error:
        raise EnvironmentError_('fixed GBDT public checker unavailable') from error


def gbdt_fixture(csv_path=None, loader_path=None):
    """200 public training rows queried against themselves; no quality gate."""
    package = Path(__file__).absolute().parent.parent
    csv_path = Path(csv_path) if csv_path is not None else package / 'data/training/gbdt/train.csv'
    loader_path = Path(loader_path) if loader_path is not None else package / 'examples/load_gbdt_training.py'
    check = _gbdt_check()
    try:
        features, labels = check.load_training(csv_path, loader_path)
        payload = check.encode_training_smoke(features, labels).decode('ascii')
    except (OSError, ValueError) as error:
        raise EnvironmentError_('fixed public GBDT training data unavailable') from error
    return payload, {'prediction_count': 200, 'diagnostic_kind': 'TRAINING_SELF_QUERY_SHAPE_DOMAIN_ONLY'}


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
            'gbdt': gbdt_fixture, 'kmeans': clustering_fixture, 'hmm': hmm_fixture}


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
    executable_name = 'selftest-driver.exe' if platform.system() == 'Windows' else 'selftest-driver'
    executable = Path(workdir) / executable_name
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
    if not executable.is_file():
        raise EnvironmentError_(f'compiler did not produce executable: {executable}')
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
            try:
                process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE,
                                           stdout=out, stderr=err, **kwargs)
            except OSError as error:
                raise EnvironmentError_(f'could not start executable: {error}') from error
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
    if task == 'gbdt':
        if spec.get('prediction_count') != 200:
            raise InvalidOutput('GBDT requires fixed 200-row public self-query')
        try:
            return _gbdt_check().parse_predictions(text.encode('ascii'))
        except (ValueError, UnicodeEncodeError) as error:
            raise InvalidOutput('GBDT invalid integer/count/domain output') from error
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
        prediction = _parse_vector(tokens, count, int,
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
        return True, 'training_self_query shape/domain only; metrics=NOT_RUN reference_score=null'
    if task == 'kmeans':
        score = ari(spec['truth'], prediction)
        return score > 0.8, f'ari={score:.3f}'
    exact = sum(path == want for path, want in zip(prediction, spec['expected']))
    return exact == len(spec['expected']), f'exact_sequences={exact}/{len(spec["expected"])}'


def run_task(executable, task, timeout, workdir, public_validation=None, *, metrics=None):
    payload, spec = public_validation if public_validation is not None else FIXTURES[task]()
    payload = payload.replace(TASK_TOKEN_PLACEHOLDER, task, 1)
    category, text, detail = run_binary(executable, payload, timeout, workdir)
    if category != 'ok':
        return category, detail
    try:
        prediction = parse_output(task, text, spec)
    except InvalidOutput as error:
        return 'invalid_output', str(error)
    passed, detail = quality_check(task, prediction, spec)
    return ('ok', detail) if passed else ('quality_check_failed', detail)


# ------------------------------------------------------ public validation
















def write_public_report(path, report):
    """Create a strict JSON report without replacing any existing destination."""
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    with open(path, 'x', encoding='utf-8') as handle:
        handle.write(payload)


# --------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=('Advisor public selftest13: compile and execute your own C++ locally. '
                     'NOT a sandbox or official grading. GBDT is training self-query '
                     'shape/domain only; other tasks keep their public micro-fixtures. '
                     'Official toolchain: ' + OFFICIAL_TOOLCHAIN))
    parser.add_argument('--project', default='starter')
    parser.add_argument('--compiler')
    parser.add_argument('--timeout', type=float, default=10.0)
    parser.add_argument('--compile-timeout', type=float, default=120.0)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--gbdt-training', type=Path, help='absolute fixed public training CSV')
    parser.add_argument('--gbdt-loader', type=Path, help='absolute fixed public loader')
    parser.add_argument('--public-data', help='legacy validation is deliberately unsupported')
    parser.add_argument('--report-json', help='new exclusive local diagnostic JSON, never a grade')
    args = parser.parse_args(argv)
    if args.public_data is not None:
        parser.error('--public-data legacy regression manifests are not advisor-compatible')
    if any(not math.isfinite(t) or t <= 0 for t in (args.timeout, args.compile_timeout)):
        parser.error('timeouts must be positive finite seconds')
    requested = args.tasks.split(',')
    if (len(requested) not in (4, 5) or len(set(requested)) != len(requested)
            or any(task not in TASKS for task in requested)):
        parser.error('--tasks requires four or five distinct names from ' + ','.join(TASKS))
    selected_tasks = tuple(task for task in TASKS if task in requested)
    report = {'schema': PUBLIC_SELFTEST_VERSION, 'policy_version': POLICY_VERSION,
              'selected_tasks': list(selected_tasks), 'status': 'NOT_RUN',
              'metrics': 'NOT_RUN', 'reference_score': None,
              'official_grading': False,
              'tasks': {task: {'status': 'NOT_RUN', 'metrics': 'NOT_RUN',
                               'reference_score': None,
                               'diagnostic_kind': ('TRAINING_SELF_QUERY_SHAPE_DOMAIN_ONLY'
                                                   if task == 'gbdt' else 'PUBLIC_MICRO_FIXTURE')}
                        for task in selected_tasks}}

    def finish(code):
        report['status'] = ('LOCAL_CHECKS_PASSED_NOT_A_GRADE' if code == 0
                            else 'ENVIRONMENT_ERROR' if code == 3 else 'LOCAL_CHECKS_FAILED')
        if args.report_json is not None:
            try:
                write_public_report(args.report_json, report)
            except (OSError, ValueError, OverflowError):
                print('selftest: report could not be created', file=sys.stderr)
                return 3
        return code

    print('ML-Assignment-1 advisor public selftest13')
    print('note: executes YOUR C++ locally; NOT a sandbox or official grading')
    print('note: GBDT checks shape/domain on public training self-query, NOT ML quality')
    print('note: HMM official Q=.5*state_accuracy+.5*exact_sequence_accuracy; reference=25*q')
    try:
        fixtures = {task: (gbdt_fixture(args.gbdt_training, args.gbdt_loader)
                           if task == 'gbdt' else FIXTURES[task]()) for task in selected_tasks}
        compiler = find_compiler(args.compiler)
    except EnvironmentError_ as error:
        print(f'selftest: environment error: {error}', file=sys.stderr)
        return finish(3)
    with tempfile.TemporaryDirectory(prefix='ml-selftest-') as workdir:
        try:
            executable, error = compile_project(compiler, args.project, workdir, args.compile_timeout)
        except EnvironmentError_ as error:
            print(f'selftest: environment error: {error}', file=sys.stderr)
            return finish(3)
        if executable is None:
            print('compile: FAIL compile_error')
            print((error or '').strip())
            return finish(1)
        print('compile: ok')
        failures = 0
        for task in selected_tasks:
            try:
                category, detail = run_task(executable, task, args.timeout, workdir, fixtures[task])
            except EnvironmentError_:
                report['tasks'][task]['status'] = 'environment_error'
                return finish(3)
            report['tasks'][task]['status'] = category
            print(f'task {task}: {category} {detail.replace(chr(10), " | ")[:300]}')
            failures += category != 'ok'
    print(f'local checks: {len(selected_tasks) - failures}/{len(selected_tasks)}; not a grade')
    return finish(0 if failures == 0 else 1)


if __name__ == '__main__':
    sys.exit(main())
