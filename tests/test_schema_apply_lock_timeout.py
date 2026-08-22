"""Regression coverage for the 2026-08-22 orion-athena-hub startup hang.

`apply_memory_cards_schema` / `apply_memory_crystallizations_schema` run
synchronous DDL during FastAPI startup. Without a lock_timeout/
statement_timeout, a conflicting lock held by another session (a COPY
export, a concurrent migration) blocks the DDL -- and the whole process --
indefinitely. Live incident: ~9.5 min 502 outage on
https://athena.tail348bbe.ts.net/, root-caused via docker logs +
pg_stat_activity lock inspection.

Two layers:
  - Fast, always-on unit tests that patch psycopg2.connect and assert the
    lock_timeout/statement_timeout options are actually passed.
  - A real-Postgres integration test (skipped without RECALL_PG_DSN,
    matching the existing pattern in test_memory_cards_reverse_history.py)
    that holds a competing ACCESS EXCLUSIVE lock in one session and confirms
    the schema-apply call raises quickly instead of hanging.
"""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fast unit tests: connect() must be called with lock_timeout/statement_timeout.
# ---------------------------------------------------------------------------


def _fake_connect_cm():
    """Build a MagicMock usable as `with psycopg2.connect(...) as conn:`."""
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False
    conn.cursor.return_value = cur
    return conn


def test_apply_memory_cards_schema_sets_lock_and_statement_timeout():
    from orion.core.storage import memory_cards

    fake_conn = _fake_connect_cm()
    with patch.object(memory_cards, "psycopg2") as mock_psycopg2:
        mock_psycopg2.connect.return_value = fake_conn
        memory_cards.apply_memory_cards_schema("postgresql://example/db")

    assert mock_psycopg2.connect.call_count == 1
    _, kwargs = mock_psycopg2.connect.call_args
    assert "options" in kwargs, "psycopg2.connect must be called with lock/statement timeout options"
    options = kwargs["options"]
    assert f"lock_timeout={memory_cards._SCHEMA_APPLY_LOCK_TIMEOUT_MS}" in options
    assert f"statement_timeout={memory_cards._SCHEMA_APPLY_STATEMENT_TIMEOUT_MS}" in options
    # Sanity: constants are real, positive, bounded values -- not accidentally 0/None.
    assert 0 < memory_cards._SCHEMA_APPLY_LOCK_TIMEOUT_MS <= 60_000
    assert 0 < memory_cards._SCHEMA_APPLY_STATEMENT_TIMEOUT_MS <= 120_000


def test_apply_memory_crystallizations_schema_sets_lock_and_statement_timeout():
    from orion.memory.crystallization import repository

    fake_conn = _fake_connect_cm()
    with patch("psycopg2.connect", return_value=fake_conn) as mock_connect:
        repository.apply_memory_crystallizations_schema("postgresql://example/db")

    assert mock_connect.call_count == 1
    _, kwargs = mock_connect.call_args
    assert "options" in kwargs, "psycopg2.connect must be called with lock/statement timeout options"
    options = kwargs["options"]
    assert f"lock_timeout={repository._SCHEMA_APPLY_LOCK_TIMEOUT_MS}" in options
    assert f"statement_timeout={repository._SCHEMA_APPLY_STATEMENT_TIMEOUT_MS}" in options
    assert 0 < repository._SCHEMA_APPLY_LOCK_TIMEOUT_MS <= 60_000
    assert 0 < repository._SCHEMA_APPLY_STATEMENT_TIMEOUT_MS <= 120_000


# ---------------------------------------------------------------------------
# Real-Postgres integration tests: prove the DDL actually fails fast under
# real lock contention instead of hanging.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("RECALL_PG_DSN"), reason="RECALL_PG_DSN not set")
def test_apply_memory_cards_schema_fails_fast_on_lock_contention() -> None:
    pytest.importorskip("psycopg2")
    import psycopg2
    import psycopg2.errors

    from orion.core.storage.memory_cards import (
        _SCHEMA_APPLY_LOCK_TIMEOUT_MS,
        apply_memory_cards_schema,
    )

    dsn = os.environ["RECALL_PG_DSN"]

    # Ensure the table exists first (outside the lock window) so the
    # contention below is on ALTER/CREATE INDEX IF NOT EXISTS statements,
    # not the initial CREATE TABLE.
    apply_memory_cards_schema(dsn)

    blocker = psycopg2.connect(dsn)
    blocker.autocommit = False
    result: dict = {}

    try:
        with blocker.cursor() as bcur:
            bcur.execute("LOCK TABLE memory_cards IN ACCESS EXCLUSIVE MODE")

            def _run():
                start = time.monotonic()
                try:
                    apply_memory_cards_schema(dsn)
                    result["elapsed"] = time.monotonic() - start
                    result["error"] = None
                except Exception as exc:  # noqa: BLE001 - capturing for assertion
                    result["elapsed"] = time.monotonic() - start
                    result["error"] = exc

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            # Bounded wait: lock_timeout is 10s: give real headroom but fail
            # the test (not hang the suite) if something regresses.
            t.join(timeout=(_SCHEMA_APPLY_LOCK_TIMEOUT_MS / 1000.0) + 15.0)
            assert not t.is_alive(), (
                "apply_memory_cards_schema did not return within lock_timeout + headroom "
                "-- the DDL call is hanging on lock contention again"
            )
    finally:
        blocker.rollback()
        blocker.close()

    assert result.get("error") is not None, "expected a lock-timeout error, got success"
    assert isinstance(result["error"], psycopg2.errors.LockNotAvailable), (
        f"expected LockNotAvailable, got {type(result['error'])}: {result['error']}"
    )
    # Should fail close to lock_timeout (10s), not hang for e.g. minutes.
    assert result["elapsed"] < (_SCHEMA_APPLY_LOCK_TIMEOUT_MS / 1000.0) + 10.0


@pytest.mark.skipif(not os.environ.get("RECALL_PG_DSN"), reason="RECALL_PG_DSN not set")
def test_apply_memory_crystallizations_schema_fails_fast_on_lock_contention() -> None:
    pytest.importorskip("psycopg2")
    import psycopg2
    import psycopg2.errors

    from orion.memory.crystallization.repository import (
        _SCHEMA_APPLY_LOCK_TIMEOUT_MS,
        apply_memory_crystallizations_schema,
    )

    dsn = os.environ["RECALL_PG_DSN"]
    apply_memory_crystallizations_schema(dsn)

    blocker = psycopg2.connect(dsn)
    blocker.autocommit = False
    result: dict = {}

    try:
        with blocker.cursor() as bcur:
            bcur.execute("LOCK TABLE memory_crystallizations IN ACCESS EXCLUSIVE MODE")

            def _run():
                start = time.monotonic()
                try:
                    apply_memory_crystallizations_schema(dsn)
                    result["elapsed"] = time.monotonic() - start
                    result["error"] = None
                except Exception as exc:  # noqa: BLE001
                    result["elapsed"] = time.monotonic() - start
                    result["error"] = exc

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=(_SCHEMA_APPLY_LOCK_TIMEOUT_MS / 1000.0) + 15.0)
            assert not t.is_alive(), (
                "apply_memory_crystallizations_schema did not return within lock_timeout + headroom "
                "-- the DDL call is hanging on lock contention again"
            )
    finally:
        blocker.rollback()
        blocker.close()

    assert result.get("error") is not None, "expected a lock-timeout error, got success"
    assert isinstance(result["error"], psycopg2.errors.LockNotAvailable), (
        f"expected LockNotAvailable, got {type(result['error'])}: {result['error']}"
    )
    assert result["elapsed"] < (_SCHEMA_APPLY_LOCK_TIMEOUT_MS / 1000.0) + 10.0
