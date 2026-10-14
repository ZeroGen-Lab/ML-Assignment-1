"""Explicit self-report request provider (identity decision 2026-10-10).

make_request(base_url, *, environ=None, timeout=30) returns the request
callable expected by student_tools.client, wiring the existing HTTPS
transport with a provider built from the three exported values
STUDENT_ID / STUDENT_NAME / STUDENT_NICKNAME. ``environ`` defaults to
os.environ and is snapshotted once per request via ``get`` on the three
keys only; the snapshot is checked with identity.check_config (the same
pure shape rule the service applies). A missing or misshaped value fails
closed with ClientError('SERVICE_NOT_CONFIGURED') before any connection is
opened, and the value itself is never echoed.

The three values travel as metadata under fixed headers, canonical base64
of their UTF-8 encoding, so raw non-ASCII never enters an ASCII header.
Self-report is attribution, not authentication; no new token or variable
is introduced.
"""
import base64
import os

from student_tools.client import ClientError
from student_tools.identity import FIELDS, check_config
from student_tools import transport

_HEADERS = {'STUDENT_ID': 'X-Self-Reported-Id',
            'STUDENT_NAME': 'X-Self-Reported-Name',
            'STUDENT_NICKNAME': 'X-Self-Reported-Nickname'}


def _encode(value):
    return base64.b64encode(value.encode('utf-8')).decode('ascii')


def make_request(base_url, *, environ=None, timeout=30):
    """Return the self-report request callable for ``base_url``."""
    source = os.environ if environ is None else environ

    def get_headers():
        snapshot = {field: source.get(field) for field in FIELDS}
        if not check_config(snapshot)['configured']:
            raise ClientError('SERVICE_NOT_CONFIGURED')
        return {_HEADERS[field]: _encode(snapshot[field]) for field in FIELDS}

    return transport.make_request(base_url, get_headers=get_headers, timeout=timeout)
