"""Tests for ``_back_populate_chat_spark_meta_from_telemetry`` in
``services/orion-sql-writer/app/worker.py`` -- the "Bi-Directional Metadata
Sync" step that writes SparkTelemetrySQL data back into the corresponding
chat_history_log row's spark_meta.

Extracted to its own function (code review on the substrate-purge branch,
2026-08-19) after finding it was not routing-aware: it only ever queried
ChatHistoryLogSQL, so telemetry for a correlation_id already routed to
aitown_chat_history_log (PR #1743) silently back-populated nothing -- no
error raised, just a missed spark_meta merge.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_telemetry_backfill_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _compiled_sql(stmt) -> str:
    from sqlalchemy.dialects import postgresql

    return str(stmt.compile(dialect=postgresql.dialect())).lower()


class _Row:
    def __init__(self, spark_meta: dict | None) -> None:
        self.spark_meta = spark_meta


class _FakeSession:
    def __init__(self, *, rows_by_model: dict[type, Any] | None = None):
        self.executed: list[Any] = []
        self.lock_calls: list[Any] = []
        self.query_calls: list[type] = []
        self.committed = False
        self._rows_by_model = rows_by_model or {}
        self._current_model: type | None = None

    def execute(self, stmt, params=None):
        if params is not None:
            self.lock_calls.append((stmt, params))
            return None
        self.executed.append(stmt)

    def commit(self):
        self.committed = True

    def query(self, model):
        self.query_calls.append(model)
        self._current_model = model
        return self

    def filter(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def with_for_update(self):
        return self

    def first(self):
        return self._rows_by_model.get(self._current_model)


def test_backfills_primary_table_when_row_found_there(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)
    sess = _FakeSession(rows_by_model={worker.ChatHistoryLogSQL: _Row({"existing": True})})

    updated = worker._back_populate_chat_spark_meta_from_telemetry(
        sess, "corr-1", {"correlation_id": "corr-1"}
    )

    assert updated is True
    assert sess.query_calls == [worker.ChatHistoryLogSQL]  # found on first try, no mirror lookup
    assert sess.committed is True
    sql = _compiled_sql(sess.executed[0])
    assert "update chat_history_log" in sql


def test_falls_back_to_mirror_table_when_primary_misses(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)
    sess = _FakeSession(
        rows_by_model={
            worker.ChatHistoryLogSQL: None,
            worker.AitownChatHistoryLogSQL: _Row({"existing": True}),
        }
    )

    updated = worker._back_populate_chat_spark_meta_from_telemetry(
        sess, "aitown-corr-1", {"correlation_id": "aitown-corr-1"}
    )

    assert updated is True
    assert sess.query_calls == [worker.ChatHistoryLogSQL, worker.AitownChatHistoryLogSQL]
    sql = _compiled_sql(sess.executed[0])
    assert "update aitown_chat_history_log" in sql


def test_does_not_check_mirror_table_when_routing_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", False)
    sess = _FakeSession(
        rows_by_model={
            worker.ChatHistoryLogSQL: None,
            worker.AitownChatHistoryLogSQL: _Row({"existing": True}),
        }
    )

    updated = worker._back_populate_chat_spark_meta_from_telemetry(
        sess, "aitown-corr-2", {"correlation_id": "aitown-corr-2"}
    )

    assert updated is False
    assert sess.query_calls == [worker.ChatHistoryLogSQL]  # no fallback attempted
    assert sess.executed == []


def test_returns_false_when_missing_from_both_tables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)
    sess = _FakeSession(
        rows_by_model={worker.ChatHistoryLogSQL: None, worker.AitownChatHistoryLogSQL: None}
    )

    updated = worker._back_populate_chat_spark_meta_from_telemetry(
        sess, "corr-missing", {"correlation_id": "corr-missing"}
    )

    assert updated is False
    assert sess.executed == []


def test_acquires_the_advisory_lock_before_any_lookup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)
    sess = _FakeSession(rows_by_model={worker.ChatHistoryLogSQL: _Row({})})

    worker._back_populate_chat_spark_meta_from_telemetry(sess, "corr-lock", {"correlation_id": "corr-lock"})

    assert len(sess.lock_calls) == 1
    lock_sql = str(sess.lock_calls[0][0]).lower()
    assert "pg_advisory_xact_lock" in lock_sql
    assert sess.lock_calls[0][1] == {"row_id": "corr-lock"}


def test_does_not_lock_when_routing_disabled(monkeypatch: pytest.MonkeyPatch):
    """Review follow-up (2026-08-19): the advisory lock exists solely to
    protect the cross-table routing decision below -- with routing off there
    is no cross-table race, so this hot-path telemetry write skips the lock
    round trip entirely rather than paying for it unconditionally."""
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", False)
    sess = _FakeSession(rows_by_model={worker.ChatHistoryLogSQL: _Row({})})

    worker._back_populate_chat_spark_meta_from_telemetry(sess, "corr-nolock", {"correlation_id": "corr-nolock"})

    assert sess.lock_calls == []
