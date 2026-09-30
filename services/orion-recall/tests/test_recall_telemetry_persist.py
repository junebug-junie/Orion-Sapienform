"""recall_telemetry persistence: before 2026-09-29 every insert raised
"can't adapt type 'dict'" (raw dict/list bound to jsonb columns) and the
failure was logged at debug, so the table never received a row."""

from __future__ import annotations

import asyncio
import threading

import psycopg2
import psycopg2.extensions

from app import worker
from orion.core.contracts.recall import RecallDecisionV1


class _AdaptingCursor:
    """Runs every bound parameter through psycopg2's real adapter, the step
    that raised in production, without needing a database. ``existing`` are
    the columns information_schema reports; ``alter_error`` makes ALTER raise
    (e.g. a lock_timeout behind a backup)."""

    existing: set = set()
    alter_error: Exception | None = None

    def __init__(self, log):
        self.log = log
        self._last = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        for p in params or ():
            psycopg2.extensions.adapt(p).getquoted()
        self.log.append((sql, params, threading.current_thread()))
        self._last = sql
        if "ALTER TABLE" in sql and _AdaptingCursor.alter_error is not None:
            raise _AdaptingCursor.alter_error

    def fetchall(self):
        if "information_schema.columns" in self._last:
            return [(c,) for c in sorted(_AdaptingCursor.existing)]
        return []


class _FakeConn:
    def __init__(self, log):
        self.log = log
        self.autocommit = False

    def cursor(self):
        return _AdaptingCursor(self.log)

    def close(self):
        pass


def _install(monkeypatch):
    log: list = []
    monkeypatch.setattr(worker.settings, "RECALL_PG_DSN", "postgresql://fake/db")
    monkeypatch.setattr(worker.psycopg2, "connect", lambda dsn, **kw: (log.append(("CONNECT", kw, None)), _FakeConn(log))[1])
    monkeypatch.setattr(worker, "_telemetry_table_ready", False)
    monkeypatch.setattr(worker, "_telemetry_failure_warned", False)
    monkeypatch.setattr(worker, "_telemetry_present_columns", set())
    monkeypatch.setattr(_AdaptingCursor, "existing", set())
    monkeypatch.setattr(_AdaptingCursor, "alter_error", None)
    return log


def _decision() -> RecallDecisionV1:
    return RecallDecisionV1(
        corr_id="c1",
        query="what did we talk about",
        selected_ids=["a", "b"],
        backend_counts={"vector": 3, "sql_chat": 1},
        latency_ms=812,
    )


def test_insert_binds_json_columns_psycopg2_can_adapt(monkeypatch) -> None:
    log = _install(monkeypatch)
    worker._persist_decision(_decision())
    inserts = [e for e in log if "INSERT INTO recall_telemetry" in e[0]]
    assert len(inserts) == 1
    assert inserts[0][1][9] == 812


def test_create_table_runs_once_per_process(monkeypatch) -> None:
    log = _install(monkeypatch)
    worker._persist_decision(_decision())
    worker._persist_decision(_decision())
    assert sum("CREATE TABLE" in e[0] for e in log) == 1
    assert sum("INSERT INTO" in e[0] for e in log) == 2


def test_async_wrapper_writes_off_the_event_loop_thread(monkeypatch) -> None:
    log = _install(monkeypatch)
    asyncio.run(worker.persist_decision_async(_decision()))
    assert log and all(e[2] is not threading.main_thread() for e in log)


def test_first_failure_is_a_warning(monkeypatch, caplog) -> None:
    _install(monkeypatch)

    def _boom(dsn, **kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(worker.psycopg2, "connect", _boom)
    with caplog.at_level("DEBUG", logger=worker.logger.name):
        worker._persist_decision(_decision())
        worker._persist_decision(_decision())
    levels = [r.levelname for r in caplog.records if "recall_telemetry_persist_failed" in r.getMessage()]
    assert levels == ["WARNING", "DEBUG"]


def test_connect_is_time_bounded(monkeypatch) -> None:
    log = _install(monkeypatch)
    worker._persist_decision(_decision())
    assert log[0] == ("CONNECT", {"connect_timeout": 3}, None)


def test_bounded_retrieval_columns_added_once_and_written(monkeypatch) -> None:
    log = _install(monkeypatch)
    decision = _decision().model_copy(
        update={
            "query_chars": 42,
            "retrieval_query_source": "condensed",
            "sub_query_count": 5,
            "candidates_fetched": 30,
            "candidates_kept": 12,
            "deadline_hit": True,
            "timings_ms": {"intake": 1, "total": 900},
        }
    )
    worker._persist_decision(decision)
    worker._persist_decision(decision)
    alters = [e[0] for e in log if "ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS" in e[0]]
    # Once per process, one per column.
    assert len(alters) == len(worker._TELEMETRY_BOUNDED_RETRIEVAL_COLUMNS)
    for col in ("query_chars", "retrieval_query_source", "sub_query_count", "candidates_fetched",
                "candidates_kept", "deadline_hit", "timings_ms"):
        assert any(col in a for a in alters)
    insert = [e for e in log if "INSERT INTO recall_telemetry" in e[0]][0]
    params = insert[1]
    assert params[10:16] == (42, "condensed", 5, 30, 12, True)
    assert params[16].adapted == {"intake": 1, "total": 900}


def test_old_decision_without_new_fields_writes_nulls(monkeypatch) -> None:
    log = _install(monkeypatch)
    worker._persist_decision(_decision())
    params = [e for e in log if "INSERT INTO recall_telemetry" in e[0]][0][1]
    assert params[10:16] == (None, None, None, None, None, None)


def test_sql_file_declares_every_bounded_retrieval_column() -> None:
    from pathlib import Path

    sql = (Path(__file__).resolve().parents[1] / "sql" / "recall_telemetry.sql").read_text()
    for ddl in worker._TELEMETRY_BOUNDED_RETRIEVAL_COLUMNS:
        assert f"ADD COLUMN IF NOT EXISTS {ddl};" in sql


_NEW_COLS = ("query_chars", "retrieval_query_source", "sub_query_count", "candidates_fetched",
             "candidates_kept", "deadline_hit", "timings_ms")


def test_existing_columns_take_no_alter_lock(monkeypatch) -> None:
    """Review, PR #2416: ADD COLUMN IF NOT EXISTS takes ACCESS EXCLUSIVE even
    when the column exists; with every column already present no ALTER runs."""
    log = _install(monkeypatch)
    monkeypatch.setattr(_AdaptingCursor, "existing", set(_NEW_COLS) | {"id", "query"})
    worker._persist_decision(_decision())
    assert not any("ALTER TABLE" in e[0] for e in log)
    insert = [e for e in log if "INSERT INTO recall_telemetry" in e[0]][0]
    assert all(c in insert[0] for c in _NEW_COLS)


def test_only_missing_columns_are_altered_after_lock_timeout_is_set(monkeypatch) -> None:
    log = _install(monkeypatch)
    monkeypatch.setattr(_AdaptingCursor, "existing", {"query_chars", "timings_ms"})
    worker._persist_decision(_decision())
    sqls = [e[0] for e in log]
    alters = [q for q in sqls if "ALTER TABLE" in q]
    assert len(alters) == len(_NEW_COLS) - 2
    assert not any("query_chars" in a or "timings_ms" in a for a in alters)
    first_alter = sqls.index(alters[0])
    assert any("lock_timeout" in q for q in sqls[:first_alter])
    assert any("statement_timeout" in q for q in sqls[:first_alter])


def test_ddl_lock_timeout_does_not_block_insert_or_retry(monkeypatch) -> None:
    log = _install(monkeypatch)
    monkeypatch.setattr(_AdaptingCursor, "alter_error", RuntimeError("canceling statement due to lock timeout"))
    worker._persist_decision(_decision())
    worker._persist_decision(_decision())
    inserts = [e for e in log if "INSERT INTO recall_telemetry" in e[0]]
    assert len(inserts) == 2
    # No confirmed new columns -> the pre-2026-09-29 row shape, still written.
    assert not any(c in inserts[0][0] for c in _NEW_COLS)
    assert len(inserts[0][1]) == 10
    # DDL attempted once per process, not per request.
    assert sum("ALTER TABLE" in e[0] for e in log) == 1
    assert sum("information_schema.columns" in e[0] for e in log) == 1
