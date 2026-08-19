"""Tests for ``_fetch_chat_turn_for_memory_emit`` in
``services/orion-sql-writer/app/worker.py``.

Regression coverage for a 4th primary-table-only blind spot found by code
review (2026-08-19) in the same pass that fixed ``_apply_spark_meta_patch``,
``_back_populate_chat_spark_meta_from_telemetry``, and
``_chat_history_thought_for_merge``: before this fix, a turn already routed
to ``aitown_chat_history_log`` (PR #1743) silently dropped out of memory
consolidation -- this function returned ``None``, its caller
(``_maybe_emit_memory_turn_from_row``) just returned, no log, no error.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_memory_emit_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class _Row:
    def __init__(self, **kwargs):
        self.prompt = kwargs.get("prompt", "hello")
        self.response = kwargs.get("response", "hi there")
        self.spark_meta = kwargs.get("spark_meta", {})
        self.session_id = kwargs.get("session_id", "sess-1")
        self.client_meta = kwargs.get("client_meta", None)


class _FakeQuery:
    def __init__(self, row):
        self._row = row

    def filter(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def first(self):
        return self._row


class _FakeSession:
    def __init__(self, *, rows_by_model: dict[type, object] | None = None):
        self._rows_by_model = rows_by_model or {}
        self.query_calls: list[type] = []
        self.closed = False

    def query(self, model):
        self.query_calls.append(model)
        return _FakeQuery(self._rows_by_model.get(model))

    def close(self):
        self.closed = True


def _patch_session(monkeypatch, sess):
    monkeypatch.setattr(worker, "get_session", lambda: sess)
    monkeypatch.setattr(worker, "remove_session", lambda: None)


def test_returns_none_for_empty_corr_id(monkeypatch: pytest.MonkeyPatch):
    assert worker._fetch_chat_turn_for_memory_emit("") is None
    assert worker._fetch_chat_turn_for_memory_emit(None) is None


def test_finds_row_in_primary_table(monkeypatch: pytest.MonkeyPatch):
    sess = _FakeSession(rows_by_model={worker.ChatHistoryLogSQL: _Row()})
    _patch_session(monkeypatch, sess)
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)

    turn = worker._fetch_chat_turn_for_memory_emit("corr-1")

    assert turn is not None
    assert turn["correlation_id"] == "corr-1"
    assert sess.query_calls == [worker.ChatHistoryLogSQL]  # no mirror lookup needed


def test_falls_back_to_mirror_table_when_primary_misses(monkeypatch: pytest.MonkeyPatch):
    sess = _FakeSession(
        rows_by_model={
            worker.ChatHistoryLogSQL: None,
            worker.AitownChatHistoryLogSQL: _Row(prompt="ai-town prompt", response="ai-town response"),
        }
    )
    _patch_session(monkeypatch, sess)
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)

    turn = worker._fetch_chat_turn_for_memory_emit("aitown-corr-1")

    assert turn is not None
    assert turn["prompt"] == "ai-town prompt"
    assert sess.query_calls == [worker.ChatHistoryLogSQL, worker.AitownChatHistoryLogSQL]


def test_does_not_check_mirror_table_when_routing_disabled(monkeypatch: pytest.MonkeyPatch):
    sess = _FakeSession(
        rows_by_model={
            worker.ChatHistoryLogSQL: None,
            worker.AitownChatHistoryLogSQL: _Row(),
        }
    )
    _patch_session(monkeypatch, sess)
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", False)

    turn = worker._fetch_chat_turn_for_memory_emit("corr-2")

    assert turn is None
    assert sess.query_calls == [worker.ChatHistoryLogSQL]  # no fallback attempted


def test_returns_none_when_missing_from_both_tables(monkeypatch: pytest.MonkeyPatch):
    sess = _FakeSession(rows_by_model={worker.ChatHistoryLogSQL: None, worker.AitownChatHistoryLogSQL: None})
    _patch_session(monkeypatch, sess)
    monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)

    turn = worker._fetch_chat_turn_for_memory_emit("corr-missing")

    assert turn is None
