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
                    # force: the fingerprint skip would otherwise (correctly)
                    # bypass the DDL this test exists to contend.
                    apply_memory_cards_schema(dsn, force=True)
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
                    apply_memory_crystallizations_schema(dsn, force=True)
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


# ---------------------------------------------------------------------------
# 2026-10-14: a boot during the nightly pg_dump hit lock_timeout on a no-op
# ALTER ... ADD COLUMN IF NOT EXISTS. A current schema must skip its DDL.
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, stored):
        self.stored = stored
        self.executed = []
        self._next = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if sql.startswith("SELECT to_regclass"):
            self._next = ("orion_schema_fingerprints",) if self.stored is not None else (None,)
        elif sql.startswith("SELECT sha256"):
            self._next = (self.stored,) if self.stored else None
        else:
            self._next = None

    def fetchone(self):
        return self._next


class _FakeConn:
    def __init__(self, cur):
        self.cur = cur
        self.autocommit = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self.cur


def _apply(stored, *, force=False, sql="CREATE TABLE IF NOT EXISTS t (x int);"):
    from orion.core.storage.schema_fingerprint import apply_schema_file_once

    cur = _FakeCursor(stored)
    ran = apply_schema_file_once(
        "postgresql://example/db", name="memory_cards", sql=sql,
        lock_timeout_ms=10_000, statement_timeout_ms=30_000,
        connect=lambda dsn, options: _FakeConn(cur), force=force,
    )
    return ran, [q for q, _ in cur.executed]


def test_current_fingerprint_skips_the_ddl():
    from orion.core.storage.schema_fingerprint import sql_fingerprint

    sql = "ALTER TABLE memory_cards ADD COLUMN IF NOT EXISTS x int;"
    ran, executed = _apply(sql_fingerprint(sql), sql=sql)
    assert ran is False
    assert sql not in executed
    assert not any(q.startswith("INSERT") for q in executed)


def test_changed_file_applies_then_records_fingerprint():
    sql = "ALTER TABLE memory_cards ADD COLUMN IF NOT EXISTS y int;"
    ran, executed = _apply("stale-hash", sql=sql)
    assert ran is True
    assert executed.index(sql) < next(i for i, q in enumerate(executed) if q.startswith("INSERT"))


def test_first_boot_without_fingerprint_table_applies():
    ran, executed = _apply(None)
    assert ran is True
    assert any(q.startswith("CREATE TABLE IF NOT EXISTS orion_schema_fingerprints") for q in executed)


def test_force_applies_even_when_current():
    from orion.core.storage.schema_fingerprint import sql_fingerprint

    sql = "SELECT 1;"
    ran, executed = _apply(sql_fingerprint(sql), sql=sql, force=True)
    assert ran is True and sql in executed


def test_failed_ddl_never_records_fingerprint():
    from orion.core.storage.schema_fingerprint import apply_schema_file_once

    class _Boom(_FakeCursor):
        def execute(self, sql, params=None):
            super().execute(sql, params)
            if sql.startswith("ALTER"):
                raise RuntimeError("canceling statement due to lock timeout")

    cur = _Boom(None)
    with pytest.raises(RuntimeError):
        apply_schema_file_once(
            "postgresql://example/db", name="memory_cards", sql="ALTER TABLE t ADD COLUMN IF NOT EXISTS z int;",
            lock_timeout_ms=10_000, statement_timeout_ms=30_000,
            connect=lambda dsn, options: _FakeConn(cur),
        )
    assert not any(q.startswith("INSERT") for q, _ in cur.executed)


@pytest.mark.skipif(not os.environ.get("RECALL_PG_DSN"), reason="RECALL_PG_DSN not set")
def test_current_schema_boots_through_a_backup_style_lock() -> None:
    """Real Postgres: with the schema current, a session holding ACCESS SHARE
    on memory_cards (what pg_dump holds) and another holding ACCESS EXCLUSIVE
    must not stall or fail the boot-time apply -- it is skipped outright."""
    pytest.importorskip("psycopg2")
    import psycopg2

    from orion.core.storage.memory_cards import apply_memory_cards_schema

    dsn = os.environ["RECALL_PG_DSN"]
    apply_memory_cards_schema(dsn, force=True)  # records the fingerprint
    blocker = psycopg2.connect(dsn)
    blocker.autocommit = False
    try:
        with blocker.cursor() as bcur:
            bcur.execute("LOCK TABLE memory_cards IN ACCESS EXCLUSIVE MODE")
            start = time.monotonic()
            assert apply_memory_cards_schema(dsn) is False
            assert time.monotonic() - start < 3.0
    finally:
        blocker.rollback()
        blocker.close()
