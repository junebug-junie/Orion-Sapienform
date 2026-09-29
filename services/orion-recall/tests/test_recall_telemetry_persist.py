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
    that raised in production, without needing a database."""

    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        for p in params or ():
            psycopg2.extensions.adapt(p).getquoted()
        self.log.append((sql, params, threading.current_thread()))


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
    assert inserts[0][1][-1] == 812


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
