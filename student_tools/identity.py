"""Offline shape check for the three student identity environment variables.

    python3 -B -m student_tools.identity

This is a LOCAL_CONFIG_ONLY convenience check: it verifies that STUDENT_ID,
STUDENT_NAME and STUDENT_NICKNAME are exported with a plausible shape before
you package or (later) submit. It never reads any other environment key,
never writes files, never touches the network, never imports the course
service modules and never compiles anything.

Values are never returned, stored or echoed: the report carries only a
per-field status. A passing check ("configured": true, exit 0) means the
local configuration SHAPE is acceptable -- it is not authentication, not a
receipt, and not proof that the student id exists or is authorized. See
docs/student-config-contract.md for the fixed status vocabulary and limits.
"""
import json
import os
import sys
import unicodedata

KIND = 'LOCAL_CONFIG_ONLY'
FIELDS = ('STUDENT_ID', 'STUDENT_NAME', 'STUDENT_NICKNAME')
# Limits mirror service.ledger._text (default 128; nickname 80) locally,
# without importing the private service package.
MAX_LENGTHS = {'STUDENT_ID': 128, 'STUDENT_NAME': 128, 'STUDENT_NICKNAME': 80}
# Deliberately stricter than ledger._text locally: U+2028/U+2029 (Zl/Zp) are
# line/paragraph separators, not category C, so the ledger candidate accepts
# them; this offline check rejects them so nobody ships an identifier that
# only later trips over hidden separator handling. ID still reports
# internal_whitespace first (isspace() is True for both).
SEPARATOR_CATEGORIES = ('Zl', 'Zp')


def _status(name, value):
    """Classify one field; never returns or records the value itself."""
    if value is None:
        return 'missing'
    if not isinstance(value, str):
        return 'bad_type'
    if not value.strip():
        return 'empty'
    if len(value) > MAX_LENGTHS[name]:
        return 'too_long'
    if value != value.strip():
        return 'leading_trailing_whitespace'
    if name == 'STUDENT_ID' and any(char.isspace() for char in value):
        return 'internal_whitespace'
    for char in value:
        category = unicodedata.category(char)
        if category.startswith('C') or category in SEPARATOR_CATEGORIES:
            return 'bad_character'
    return 'ok'


def check_config(environ):
    """Return a closed shape report for the three identity variables.

    ``environ`` is any mapping (os.environ or a test fixture); only the three
    FIELDS keys are read, via ``get`` only.
    """
    fields = {name: _status(name, environ.get(name)) for name in FIELDS}
    return {'kind': KIND,
            'configured': all(status == 'ok' for status in fields.values()),
            'authenticated': False,
            'fields': fields}


USAGE_ERROR = ('student_tools.identity: error: this tool takes no arguments; '
               'identity is read only from the STUDENT_ID, STUDENT_NAME and '
               'STUDENT_NICKNAME environment variables\n')
HELP_TEXT = """usage: python3 -B -m student_tools.identity

Offline shape check for STUDENT_ID, STUDENT_NAME and STUDENT_NICKNAME.
Prints one JSON report to stdout; exit 0 means the local configuration
shape is acceptable (LOCAL_CONFIG_ONLY). This is not authentication,
not a receipt and not a submission. The tool takes no arguments and
never echoes the exported values. See docs/student-config-contract.md.
"""


def main(argv=None):
    """Run the CLI check. ``argv=None`` reads sys.argv like a normal script.

    Unknown or positional arguments get a FIXED usage error (exit 2) and are
    never echoed back, so a student who mistakes the CLI for a place to put
    their id or name does not leak it into logs.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    if args and all(arg in ('-h', '--help') for arg in args):
        print(HELP_TEXT, end='')
        return 0
    if args:
        print(USAGE_ERROR, file=sys.stderr, end='')
        return 2
    report = check_config(os.environ)
    print(json.dumps(report, sort_keys=True))
    return 0 if report['configured'] else 1


if __name__ == '__main__':
    sys.exit(main())
