"""Import bootstrap for the orion-sql-writer test suite.

pytest imports ``conftest.py`` before collecting any test module in this
directory, which makes it the one place that can guarantee both import roots
exist for every test -- ``app.*`` (this service) and ``orion.*`` (the repo).

Why this file exists rather than the previous arrangement: 26 of the 42 test
modules hand-rolled their own ``sys.path.insert`` preamble, and the other 17
had none at all. Those 17 passed only because an *alphabetically earlier*
sibling happened to run first and mutate the interpreter-global ``sys.path``
on their behalf. Any subset that excluded the sibling failed at COLLECTION
with ``ModuleNotFoundError: No module named 'orion'`` -- and a collection
error is fatal to the whole run, not just the one file.

Confirmed live 2026-08-13:

    pytest tests/test_grammar_ledger_sql_shape.py tests/test_grammar_truth.py
    -> ModuleNotFoundError: No module named 'orion'   (both files aborted)

while each file passed on its own. The practical effect was that this suite's
pass/fail count depended on which files you asked for, so "18 failures" was
not a stable fact about the service -- a different invocation gave a different
number, and none of them were trustworthy.

Existing per-module preambles are left in place deliberately: they are
harmless (the guarded insert is idempotent), and removing 26 of them is churn
that would bury the actual fixes in this changeset. New test modules should
rely on this file and add no preamble of their own.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

# Order matters: SERVICE_ROOT first so `import app` resolves to THIS service's
# package rather than any same-named directory that a sibling service might
# expose on the path in a combined run.
for _root in (SERVICE_ROOT, REPO_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


# ---------------------------------------------------------------------------
# Database opt-in
# ---------------------------------------------------------------------------
#
# This deliberately does NOT default POSTGRES_URI to anything.
#
# An earlier version of this file defaulted it to
# postgresql://postgres:postgres@localhost:55432/conjourney -- which is
# Orion's LIVE database (5.9M grammar_events, newest row seconds old;
# 17k journal_entries). tests/grammar_integration_helpers.py:119 issues real
# DELETE statements, and test_grammar_ledger_integration real INSERTs first.
# app/settings.py's unresolvable docker-internal default
# (orion-athena-sql-db:5432) had been acting as accidental protection: a
# host-side `pytest` COULD NOT reach production. Defaulting to 55432 removed
# that protection with no opt-in and no warning, which is exactly the
# "production writes" AGENTS.md section 13 requires explicit approval for.
# Caught in review before this shipped.
#
# So: a real database is used only when the operator names one, via
# ORION_SQL_WRITER_TEST_DATABASE_URI. Anything else skips, loudly.
#
# Only ONE module actually needs a database. Verified 2026-08-13 by pointing
# the whole set at a valid host/port with a nonexistent database name -- so
# TCP connects but no query can succeed:
#
#   test_notify_attention_ack        3 passed   (builds its session from
#   test_notify_attention_escalate   2 passed    MagicMock and monkeypatches
#   test_grammar_truth              10 passed    get_session; never opens a
#   test_grammar_ledger_integration  3 ERRORS    socket)
#
# The first version of this file listed all four, over-skipping 15 tests --
# including the 6 cross-test isolation failures this branch exists to make
# visible. On a machine without the database they vanished again, and a
# genuine product regression injected into app/grammar_truth.py produced a
# byte-identical green run. A skip list is a place where real failures go to
# hide, so it holds exactly one entry and each entry is justified by
# measurement, not by the module's name.

_TEST_DB_ENV_VAR = "ORION_SQL_WRITER_TEST_DATABASE_URI"
_test_db_uri = os.environ.get(_TEST_DB_ENV_VAR)

if _test_db_uri:
    # Only now does anything point at a real database, and only because the
    # operator said so. Set both names: app/db.py builds its engines from
    # settings.postgres_uri, while some call sites read DATABASE_URL.
    os.environ["POSTGRES_URI"] = _test_db_uri
    os.environ["DATABASE_URL"] = _test_db_uri


def _database_is_reachable() -> bool:
    if not _test_db_uri:
        return False
    parsed = urlparse(_test_db_uri)
    host, port = parsed.hostname, parsed.port or 5432
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


DB_REACHABLE = _database_is_reachable()

# Measured, not guessed -- see the note above. Adding a module here must be
# justified by showing it actually fails against an unreachable database.
_DB_DEPENDENT_MODULES = {"test_grammar_ledger_integration"}

_TESTS_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items) -> None:
    if DB_REACHABLE:
        return
    skip = pytest.mark.skip(
        reason=(
            f"needs a real Postgres; set {_TEST_DB_ENV_VAR} to opt in. "
            "Use a scratch database -- these tests DELETE rows."
        )
    )
    for item in items:
        path = Path(str(item.fspath))
        # Path-scoped: pytest_collection_modifyitems receives the whole
        # session's items, so an unscoped name match would skip a same-named
        # module belonging to a different service in a combined run.
        if path.parent == _TESTS_DIR and path.stem in _DB_DEPENDENT_MODULES:
            item.add_marker(skip)
