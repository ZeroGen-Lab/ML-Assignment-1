"""Internal protocol-only client; no network/authentication or CLI configured.

An injected request callable is not evidence of service identity. Deployment
must separately establish endpoint origin and authorization. No export values
or host context are read, synthesized or transmitted here.
"""
import copy
import math
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


def _score(value, low, high):
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


def _formal_wire(result, status, selected_tasks):
    """Validate public structure, not accepted origin or a cohort pin.

    Raw fields describe this submission; cohort totals may select older
    submissions. Neither block authenticates the other or computes scores.
    """
    _keys(result, {'available', 'kind', 'policy_version', 'tasks', 'cohort'})
    _require(result['available'] is True and type(result['kind']) is str
             and result['kind'] == 'formal_result')
    _require(type(result['policy_version']) is str
             and result['policy_version'] == 'assignment-scoring-composite-2026-10-13')
    _require(type(selected_tasks) is list)
    _require(selected_tasks == list(canonical_selected_tasks(
        selected_tasks, code='INVALID_RESPONSE')))
    tasks = result['tasks']
    _keys(tasks, TASKS)
    outcomes = []
    for task in TASK_ORDER:
        slot = tasks[task]
        _keys(slot, {'outcome', 'quality', 'correctness', 'robustness', 'efficiency'})
        outcome = slot['outcome']
        _require(type(outcome) is str and outcome in (
            'NOT_SELECTED', 'measured', 'student_failure', 'no_valid_effect', 'infra_failure'))
        _require((outcome == 'NOT_SELECTED') == (task not in selected_tasks))
        if outcome == 'measured':
            _require(_score(slot['quality'], -0.6 if task == 'kmeans' else 0, 1)
                     and _score(slot['correctness'], 0, 4)
                     and _score(slot['robustness'], 0, 2)
                     and _score(slot['efficiency'], 0, 1))
        else:
            _require(all(slot[key] is None for key in (
                'quality', 'correctness', 'robustness', 'efficiency')))
        if task in selected_tasks:
            outcomes.append(outcome)
    if status == 'COMPLETED':
        _require(all(value in ('measured', 'no_valid_effect') for value in outcomes))
    elif status == 'STUDENT_FAILURE':
        _require('student_failure' in outcomes and 'infra_failure' not in outcomes)
    elif status == 'INFRASTRUCTURE_FAILURE':
        _require('infra_failure' in outcomes)
    else:
        _require(False)
    cohort = result['cohort']
    _require(type(cohort) is dict and type(cohort.get('available')) is bool)
    if not cohort['available']:
        _keys(cohort, {'available', 'code'})
        _require(type(cohort['code']) is str and cohort['code'] == 'RESULT_NOT_AVAILABLE')
        return
    _keys(cohort, {'available', 'status', 'tasks', 'assignment', 'rank'})
    _require(type(cohort['status']) is str and cohort['status'] in ('READY', 'BLOCKED'))
    _keys(cohort['tasks'], TASKS)
    if cohort['status'] == 'BLOCKED':
        _require(all(value is None for value in cohort['tasks'].values())
                 and cohort['assignment'] is None and cohort['rank'] is None)
    else:
        _require(all(_score(value, 0, 25) for value in cohort['tasks'].values())
                 and _score(cohort['assignment'], 0, 100)
                 and type(cohort['rank']) is int and 1 <= cohort['rank'] <= 500)


_TRIAL_MISSING = ('quality', 'correctness', 'robustness', 'efficiency',
                  'random_raw_points', 'cohort_raw_points')


_TRIAL_QUALITY18_POLICY = 'trial-quality18-anchors-20261018-v1'
_TRIAL_QUALITY25_POLICY = 'trial-quality25-advisor-20261020-v1'
_TRIAL_QUALITY25_SCHEMA = 'advisor-quality25-projection-v1'
_TRIAL_QUALITY25_BOUNDS = {
    'lr': (('accuracy', 0, 1), ('macro_f1', 0, 1)),
    'svm': (('accuracy', 0, 1), ('macro_f1', 0, 1)),
    'gbdt': (('accuracy', 0, 1), ('macro_f1', 0, 1)),
    'kmeans': (('ari', -1, 1), ('nmi', 0, 1)),
    'hmm': (('state_accuracy', 0, 1), ('exact_sequence_accuracy', 0, 1)),
}
_TRIAL_QUALITY25_WEIGHTS = {'lr': (0.5, 0.5), 'svm': (0.5, 0.5),
                            'gbdt': (0.5, 0.5), 'kmeans': (0.6, 0.4),
                            'hmm': (0.5, 0.5)}
_TRIAL_COMPONENTS = ('quality', 'correctness', 'robustness', 'efficiency')


def _trial_quality18_points_wire(points, selected_tasks, tasks):
    """Validate persisted-policy projection, not the effect mapping formula."""
    _require(type(points['status']) is str and points['status'] in ('READY', 'NOT_READY'))
    _keys(points['tasks'], TASKS)
    ready = []
    for task in TASK_ORDER:
        slot = points['tasks'][task]
        _keys(slot, {'status', 'raw_u', 'final_s', 'missing', 'quality',
                     'effect_s', 'correctness_s', 'robustness_s', 'efficiency_s'})
        missing = slot['missing']
        _require(type(missing) is list and all(type(key) is str for key in missing))
        _require(missing == [key for key in _TRIAL_COMPONENTS if key in missing])
        values = ('quality', 'effect_s', 'correctness_s', 'robustness_s', 'efficiency_s')
        if task not in selected_tasks:
            _require(slot['status'] == 'NOT_SELECTED' and missing == []
                     and slot['raw_u'] is None and slot['final_s'] is None
                     and all(slot[key] is None for key in values))
            continue
        readings = tasks[task]
        _require(missing == [key for key in _TRIAL_COMPONENTS if readings[key] is None])
        for source, target, low, high in (
                ('quality', 'quality', -0.6 if task == 'kmeans' else 0, 1),
                ('correctness', 'correctness_s', 0, 4),
                ('robustness', 'robustness_s', 0, 2), ('efficiency', 'efficiency_s', 0, 1)):
            value = readings[source]
            if value is None:
                _require(slot[target] is None)
            else:
                _require(_score(value, low, high) and _score(slot[target], low, high)
                         and slot[target] == value)
        if readings['quality'] is None:
            _require(slot['effect_s'] is None)
        else:
            _require(_score(slot['effect_s'], 0, 18))
        if missing:
            _require(slot['status'] == ('NOT_READY' if len(missing) == 4 else 'PARTIAL')
                     and slot['raw_u'] is None and slot['final_s'] is None)
        else:
            _require(slot['status'] == 'READY' and _score(slot['raw_u'], 0, 25)
                     and _score(slot['final_s'], 0, 25))
            total = sum(slot[key] for key in values[1:])
            _require(math.isclose(slot['raw_u'], total, rel_tol=1e-12, abs_tol=1e-9)
                     and math.isclose(slot['final_s'], total, rel_tol=1e-12, abs_tol=1e-9))
            ready.append(slot['final_s'])
    complete = len(ready) == len(selected_tasks)
    _require(points['status'] == ('READY' if complete else 'NOT_READY'))
    if complete:
        best = points['best_four']
        _require(type(best) is list and len(best) == 4
                 and all(_score(value, 0, 25) for value in best)
                 and best == sorted(ready, reverse=True)[:4])
        _require(_score(points['assignment'], 0, 100)
                 and math.isclose(points['assignment'], sum(best), rel_tol=1e-12, abs_tol=1e-9))
    else:
        _require(points['best_four'] is None and points['assignment'] is None)


def _trial_quality25_points_wire(points, selected_tasks, tasks):
    """Validate the advisor quality25 projection shape, never authority."""
    _keys(points, {'schema', 'submission_id', 'student_id', 'received_at',
                   'data_version', 'eval_config_id', 'environment_id',
                   'policy_version', 'selected_tasks', 'trial', 'provisional',
                   'formal_ready', 'tasks', 'task_score_sum_status',
                   'task_score_sum', 'course_best_four', 'course_score',
                   'teacher_final_course_score'})
    _require(points['schema'] == _TRIAL_QUALITY25_SCHEMA
             and points['policy_version'] == _TRIAL_QUALITY25_POLICY
             and points['trial'] is True and points['provisional'] is True
             and points['formal_ready'] is False)
    for field in ('submission_id', 'student_id', 'data_version',
                  'eval_config_id', 'environment_id'):
        value = points[field]
        _require(type(value) is str and 0 < len(value) <= 256
                 and not any(ord(c) < 32 or ord(c) == 127 for c in value))
    _require(type(points['received_at']) is str
             and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z',
                              points['received_at']) is not None)
    try:
        stamp = datetime.fromisoformat(points['received_at'][:-1] + '+00:00')
    except ValueError:
        _require(False)
    _require(stamp.utcoffset().total_seconds() == 0)
    _require(type(points['selected_tasks']) is list
             and points['selected_tasks'] == selected_tasks
             and points['selected_tasks'] == list(canonical_selected_tasks(
                 points['selected_tasks'], code='INVALID_RESPONSE')))
    selected = points['selected_tasks']
    _keys(points['tasks'], TASKS)
    ready = []
    for task in TASK_ORDER:
        slot = points['tasks'][task]
        _keys(slot, {'status', 'outcome', 'metrics', 'raw_quality', 'q',
                     'reference_score'})
        outcome = slot['outcome']
        _require(type(outcome) is str and outcome in (
            'measured', 'source_unknown', 'NOT_SELECTED', 'student_failure',
            'no_valid_effect', 'pending', 'infra_failure'))
        _require((outcome == 'NOT_SELECTED') == (task not in selected)
                 and outcome == tasks[task]['outcome']
                 and all(tasks[task][field] is None for field in (
                     'correctness', 'robustness', 'efficiency')))
        metrics = slot['metrics']
        _keys(metrics, {name for name, _, _ in _TRIAL_QUALITY25_BOUNDS[task]})
        for name, low, high in _TRIAL_QUALITY25_BOUNDS[task]:
            _require(metrics[name] is None or _score(metrics[name], low, high))
        if outcome == 'NOT_SELECTED':
            _require(slot['status'] == 'NOT_SELECTED' and slot['raw_quality'] is None
                     and slot['q'] is None and slot['reference_score'] is None
                     and all(value is None for value in metrics.values()))
            continue
        root_quality = tasks[task]['quality']
        if root_quality is None:
            _require(slot['raw_quality'] is None)
        else:
            _require(_score(root_quality, -0.6 if task == 'kmeans' else 0, 1)
                     and _score(slot['raw_quality'],
                                -0.6 if task == 'kmeans' else 0, 1)
                     and math.isclose(slot['raw_quality'], root_quality,
                                      rel_tol=1e-12, abs_tol=1e-12))
        if outcome not in ('measured', 'source_unknown'):
            _require(root_quality is None
                     and all(value is None for value in metrics.values()))
        if slot['raw_quality'] is not None:
            _require(all(value is not None for value in metrics.values()))
            first, second = _TRIAL_QUALITY25_WEIGHTS[task]
            names = _TRIAL_QUALITY25_BOUNDS[task]
            expected = first * metrics[names[0][0]] + second * metrics[names[1][0]]
            _require(math.isclose(slot['raw_quality'], expected,
                                  rel_tol=1e-12, abs_tol=1e-12))
        if outcome != 'measured':
            _require(slot['q'] is None and slot['reference_score'] is None
                     and slot['status'] == ('PARTIAL' if any(
                         value is not None for value in metrics.values()) else 'NOT_READY'))
            continue
        _require(_score(slot['raw_quality'], -0.6 if task == 'kmeans' else 0, 1)
                 and _score(slot['q'], 0, 1)
                 and _score(slot['reference_score'], 0, 25)
                 and math.isclose(slot['q'], min(1.0, max(0.0, slot['raw_quality'])),
                                  rel_tol=1e-12, abs_tol=1e-12)
                 and math.isclose(slot['reference_score'], 25.0 * slot['q'],
                                  rel_tol=1e-12, abs_tol=1e-9))
        _require(slot['status'] == 'READY')
        ready.append(slot['reference_score'])
    complete = len(ready) == len(selected)
    _require(points['task_score_sum_status']
             == ('COMPLETE' if complete else 'INCOMPLETE'))
    if complete:
        _require(_score(points['task_score_sum'], 0, 125)
                 and math.isclose(points['task_score_sum'], math.fsum(ready),
                                  rel_tol=1e-12, abs_tol=1e-9))
    else:
        _require(points['task_score_sum'] is None)
    if len(ready) >= 4:
        best = points['course_best_four']
        _require(type(best) is list and len(best) == 4
                 and all(_score(value, 0, 25) for value in best)
                 and best == sorted(ready, reverse=True)[:4])
        _require(_score(points['course_score'], 0, 100)
                 and math.isclose(points['course_score'], math.fsum(best),
                                  rel_tol=1e-12, abs_tol=1e-9))
    else:
        _require(points['course_best_four'] is None
                 and points['course_score'] is None)
    _require(points['teacher_final_course_score'] is None)


def _trial_points_wire(points, selected_tasks, tasks):
    """Validate the student points projection, never calculate its formula."""
    if (type(points) is dict
            and points.get('schema') == _TRIAL_QUALITY25_SCHEMA
            and points.get('policy_version') == _TRIAL_QUALITY25_POLICY):
        return _trial_quality25_points_wire(points, selected_tasks, tasks)
    _keys(points, {'status', 'policy_version', 'tasks', 'best_four', 'assignment'})
    if points['policy_version'] == _TRIAL_QUALITY18_POLICY:
        return _trial_quality18_points_wire(points, selected_tasks, tasks)
    _require(points['policy_version'] == 'assignment-scoring-composite-2026-10-13')
    _require(type(points['status']) is str and points['status'] in ('READY', 'NOT_READY'))
    _keys(points['tasks'], TASKS)
    ready = []
    for task in TASK_ORDER:
        slot = points['tasks'][task]
        _keys(slot, {'status', 'raw_u', 'final_s', 'missing'})
        missing = slot['missing']
        _require(type(missing) is list and all(type(item) is str for item in missing))
        _require(missing == [item for item in _TRIAL_MISSING if item in missing])
        if task not in selected_tasks:
            _require(slot['status'] == 'NOT_SELECTED' and slot['raw_u'] is None
                     and slot['final_s'] is None and missing == [])
            continue
        raw_missing = [key for key in _TRIAL_MISSING[:4] if tasks[task][key] is None]
        _require([key for key in missing if key in _TRIAL_MISSING[:4]] == raw_missing)
        if raw_missing:
            _require(slot['raw_u'] is None)
        else:
            _require(type(slot['raw_u']) in (int, float) and math.isfinite(slot['raw_u']))
        if missing:
            _require(slot['status'] == 'NOT_READY' and slot['final_s'] is None)
        else:
            _require(slot['status'] == 'READY' and _score(slot['final_s'], 0, 25))
            ready.append(slot['final_s'])
    complete = len(ready) == len(selected_tasks)
    _require(points['status'] == ('READY' if complete else 'NOT_READY'))
    if complete:
        best = points['best_four']
        _require(type(best) is list and len(best) == 4
                 and all(_score(value, 0, 25) for value in best)
                 and best == sorted(ready, reverse=True)[:4])
        _require(_score(points['assignment'], 0, 100)
                 and math.isclose(points['assignment'], sum(best), rel_tol=1e-12, abs_tol=1e-9))
    else:
        _require(points['best_four'] is None and points['assignment'] is None)


def _trial_wire(result, status, selected_tasks):
    """Validate trial raw readings and points without accepting a formal rank."""
    _keys(result, {'available', 'kind', 'status', 'trial', 'formal_ready',
                   'selected_tasks', 'tasks', 'error', 'points'})
    _require(result['available'] is True and result['kind'] == 'trial_result'
             and result['status'] == 'TRIAL' and result['trial'] is True
             and result['formal_ready'] is False)
    _require(status in ('COMPLETED', 'STUDENT_FAILURE', 'INFRASTRUCTURE_FAILURE'))
    _require(type(selected_tasks) is list)
    _require(selected_tasks == list(canonical_selected_tasks(
        selected_tasks, code='INVALID_RESPONSE')))
    _require(type(result['selected_tasks']) is list
             and result['selected_tasks'] == selected_tasks)
    tasks = result['tasks']
    _keys(tasks, TASKS)
    for task in TASK_ORDER:
        slot = tasks[task]
        _keys(slot, {'outcome', 'quality', 'correctness', 'robustness', 'efficiency'})
        outcome = slot['outcome']
        _require(type(outcome) is str and outcome in (
            'NOT_SELECTED', 'measured', 'source_unknown', 'student_failure',
            'no_valid_effect', 'pending', 'infra_failure'))
        _require((outcome == 'NOT_SELECTED') == (task not in selected_tasks))
        for field, low, high in (('quality', -0.6 if task == 'kmeans' else 0, 1),
                                  ('correctness', 0, 4), ('robustness', 0, 2),
                                  ('efficiency', 0, 1)):
            value = slot[field]
            if outcome not in ('measured', 'source_unknown'):
                _require(value is None)
            elif value is None:
                _require(outcome == 'source_unknown' or (
                    type(result['points']) is dict
                    and result['points'].get('policy_version') == _TRIAL_QUALITY18_POLICY)
                    or (type(result['points']) is dict
                        and result['points'].get('schema') == _TRIAL_QUALITY25_SCHEMA
                        and result['points'].get('policy_version') == _TRIAL_QUALITY25_POLICY
                        and not (outcome == 'measured' and field == 'quality')))
            else:
                _require(_score(value, low, high))
    error = result['error']
    if error is not None:
        _keys(error, {'stage', 'code', 'exit_code'})
        _require(error['stage'] == 'compile'
                 and error['code'] == 'STUDENT_COMPILE_FAILURE'
                 and type(error['exit_code']) is int and error['exit_code'] != 0
                 and status == 'STUDENT_FAILURE')
    _trial_points_wire(result['points'], selected_tasks, tasks)


def _summary(result, status, selected_tasks=None):
    """Closed legacy, trial raw or formal/cohort wire; no private metadata."""
    _require(type(result) is dict and type(result.get('available')) is bool)
    if result.get('kind') == 'trial_result':
        _trial_wire(result, status, selected_tasks)
        return
    if result.get('kind') == 'formal_result':
        _formal_wire(result, status, selected_tasks)
        return
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
    if selected_tasks is not None:
        _require(all(tasks[task]['status'] == 'not_run'
                     for task in TASKS - set(selected_tasks)))
        selected_states = [tasks[task]['status'] for task in selected_tasks]
        if status == 'COMPLETED':
            _require(failure is None and all(state == 'measured' for state in selected_states))
        elif status == 'STUDENT_FAILURE':
            if failure == 'student_compile':
                _require(all(state == 'not_run' for state in selected_states))
            elif failure == 'student_tasks':
                _require(all(state in ('measured', 'student_failure') for state in selected_states)
                         and 'student_failure' in selected_states)
            else:
                _require(False)
        elif status == 'INFRASTRUCTURE_FAILURE':
            _require(failure == 'infrastructure'
                     and any(state in ('infrastructure_failure', 'not_run')
                             for state in selected_states))
        else:
            _require(False)
        return
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


def _daily_usage(data):
    """Closed server-day projection; never infer usage from a page or local day."""
    _keys(data, {'kind', 'date', 'used', 'remaining'})
    _require(type(data['kind']) is str and data['kind'] == 'daily_usage')
    value = data['date']
    _require(type(value) is str
             and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value) is not None)
    try:
        datetime(int(value[:4]), int(value[5:7]), int(value[8:10]))
    except ValueError:
        _require(False)
    _require(type(data['used']) is int and data['used'] >= 0)
    # Wire contract: five accepted submissions per Asia/Shanghai calendar day.
    _require(type(data['remaining']) is int
             and data['remaining'] == max(0, 5 - data['used']))


def _submission(data, *, receipt=False, daily_usage=False):
    keys = COMMON | {'kind', 'status', 'attempt'}
    if type(data) is dict and 'selected_tasks' in data:
        keys |= {'selected_tasks'}
        tasks = data['selected_tasks']
        _require(type(tasks) is list)
        _require(tasks == list(canonical_selected_tasks(tasks, code='INVALID_RESPONSE')))
    if daily_usage:
        _require(not receipt)
        keys |= {'daily_usage'}
    _keys(data, keys if receipt else keys | {'result'})
    if daily_usage:
        _daily_usage(data['daily_usage'])
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
        _summary(data['result'], data['status'], data.get('selected_tasks'))


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
        _submission(data, daily_usage=True)
        _require(data['submission_id'] == fields['submission_id'])
    elif operation == 'list_submissions':
        _keys(data, {'kind', 'items', 'limit', 'offset', 'daily_usage'})
        _daily_usage(data['daily_usage'])
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
