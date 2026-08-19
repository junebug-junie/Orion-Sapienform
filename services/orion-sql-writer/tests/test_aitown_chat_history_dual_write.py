"""Tests for the AI Town chat-history table split, Phase 1
(docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md).

Covers ``_is_aitown_client_meta``, ``_maybe_dual_write_aitown_chat_history``,
and ``_maybe_dual_patch_aitown_spark_meta`` in
``services/orion-sql-writer/app/worker.py``. All DB access is faked at the
session boundary -- no real Postgres, matching the existing convention in
``test_chat_history_turn_coalesce.py`` (the regression test this design doc
itself calls out as needing to keep passing unmodified, which it does --
see that file, unchanged, still green).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_aitown_dual_write_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))
SPEC.loader.exec_module(worker)


AITOWN_CLIENT_META = {"external_room": {"platform": "aitown", "room_id": "town-square"}}
HUB_CLIENT_META = {"external_room": {"platform": "hub"}}


class _NestedCtx:
    """Fake ``Session.begin_nested()`` SAVEPOINT context manager.

    Real SQLAlchemy semantics: on a clean exit, the savepoint is released
    (no-op here); on an exception, it's rolled back and the exception
    re-raises (returning False from __exit__ does that automatically).
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, *, raise_on_execute: bool = False, query_result: Any = None):
        self.executed: list[Any] = []
        self.nested_calls = 0
        self._raise_on_execute = raise_on_execute
        self._query_result = query_result
        self.query_calls: list[Any] = []

    def execute(self, stmt):
        if self._raise_on_execute:
            raise RuntimeError("simulated mirror-table write failure")
        self.executed.append(stmt)

    def begin_nested(self):
        self.nested_calls += 1
        return _NestedCtx()

    def query(self, model_cls):
        self.query_calls.append(model_cls)
        return self

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._query_result


def _compiled_sql(stmt) -> str:
    from sqlalchemy.dialects import postgresql

    return str(stmt.compile(dialect=postgresql.dialect())).lower()


class TestIsAitownClientMeta:
    def test_matches_the_canonical_signal(self):
        assert worker._is_aitown_client_meta(AITOWN_CLIENT_META) is True

    def test_rejects_non_aitown_platform(self):
        assert worker._is_aitown_client_meta(HUB_CLIENT_META) is False

    def test_rejects_missing_external_room(self):
        assert worker._is_aitown_client_meta({"other": "stuff"}) is False

    def test_rejects_non_dict_external_room(self):
        assert worker._is_aitown_client_meta({"external_room": "not-a-dict"}) is False

    def test_rejects_non_dict_client_meta(self):
        assert worker._is_aitown_client_meta(None) is False
        assert worker._is_aitown_client_meta("garbage") is False


class TestMaybeDualWriteAitownChatHistory:
    def test_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", False)
        sess = _FakeSession()

        worker._maybe_dual_write_aitown_chat_history(
            sess, {"id": "x", "client_meta": AITOWN_CLIENT_META}, incoming_wins=True
        )

        assert sess.executed == []
        assert sess.nested_calls == 0

    def test_noop_when_not_aitown(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", True)
        sess = _FakeSession()

        worker._maybe_dual_write_aitown_chat_history(
            sess, {"id": "x", "client_meta": HUB_CLIENT_META}, incoming_wins=True
        )

        assert sess.executed == []
        assert sess.nested_calls == 0

    def test_noop_when_client_meta_absent(self, monkeypatch: pytest.MonkeyPatch):
        """Known Phase 1 limitation, documented in the function's own
        docstring: a call whose ``values`` don't carry client_meta at all
        can't be classified, and is correctly skipped rather than guessed."""
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", True)
        sess = _FakeSession()

        worker._maybe_dual_write_aitown_chat_history(sess, {"id": "x"}, incoming_wins=True)

        assert sess.executed == []

    def test_writes_to_the_mirror_table_when_aitown_and_enabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", True)
        sess = _FakeSession()

        worker._maybe_dual_write_aitown_chat_history(
            sess, {"id": "x", "correlation_id": "x", "client_meta": AITOWN_CLIENT_META}, incoming_wins=True
        )

        assert sess.nested_calls == 1
        assert len(sess.executed) == 1
        sql = _compiled_sql(sess.executed[0])
        assert "insert into aitown_chat_history_log" in sql
        assert "on conflict (id)" in sql
        assert "do update set" in sql

    def test_never_targets_the_real_table(self, monkeypatch: pytest.MonkeyPatch):
        """The one thing this function must never do: mutate chat_history_log
        itself. Same session object the caller's real upsert also uses, so
        the only thing distinguishing safety here is which table the
        compiled statement actually names."""
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", True)
        sess = _FakeSession()

        worker._maybe_dual_write_aitown_chat_history(
            sess, {"id": "x", "client_meta": AITOWN_CLIENT_META}, incoming_wins=False
        )

        sql = _compiled_sql(sess.executed[0])
        assert 'insert into aitown_chat_history_log' in sql

    def test_a_mirror_write_failure_is_contained_not_raised(self, monkeypatch: pytest.MonkeyPatch):
        """Must never take the real chat_history_log write it rides alongside
        down with it -- this must not raise even when the underlying execute
        does."""
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", True)
        sess = _FakeSession(raise_on_execute=True)

        worker._maybe_dual_write_aitown_chat_history(
            sess, {"id": "x", "client_meta": AITOWN_CLIENT_META}, incoming_wins=True
        )  # must not raise

        assert sess.nested_calls == 1

    def test_uses_a_savepoint_not_the_bare_session(self, monkeypatch: pytest.MonkeyPatch):
        """Regression guard for the transaction-abort trap documented in the
        function's own docstring: both writes share one session/transaction,
        so a mirror-write failure without a SAVEPOINT would poison the
        caller's own subsequent commit() for the real write. begin_nested()
        must be used, not a plain try/except around sess.execute alone."""
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", True)
        sess = _FakeSession()

        worker._maybe_dual_write_aitown_chat_history(
            sess, {"id": "x", "client_meta": AITOWN_CLIENT_META}, incoming_wins=True
        )

        assert sess.nested_calls == 1


class TestMaybeDualPatchAitownSparkMeta:
    def test_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", False)
        sess = _FakeSession(query_result=object())

        worker._maybe_dual_patch_aitown_spark_meta(sess, correlation_id="c1", patch_spark_meta={"a": 1})

        assert sess.executed == []
        assert sess.query_calls == []

    def test_noop_when_no_mirror_row_exists(self, monkeypatch: pytest.MonkeyPatch):
        """Patch-only, never insert: a correlation_id never dual-written
        (dual-write was off, or the row was never classified AI Town) must
        not spawn a new mirror row on the patch path's say-so."""
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", True)
        sess = _FakeSession(query_result=None)

        worker._maybe_dual_patch_aitown_spark_meta(sess, correlation_id="c1", patch_spark_meta={"a": 1})

        assert sess.query_calls == [worker.AitownChatHistoryLogSQL]
        assert sess.executed == []

    def test_patches_the_mirror_row_when_it_exists(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", True)

        class _ExistingRow:
            spark_meta = {"existing": True}

        sess = _FakeSession(query_result=_ExistingRow())

        worker._maybe_dual_patch_aitown_spark_meta(sess, correlation_id="c1", patch_spark_meta={"new": True})

        assert len(sess.executed) == 1
        sql = _compiled_sql(sess.executed[0])
        assert "update aitown_chat_history_log" in sql

    def test_a_patch_failure_is_contained_not_raised(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_dual_write_enabled", True)

        class _ExistingRow:
            spark_meta = {}

        sess = _FakeSession(query_result=_ExistingRow(), raise_on_execute=True)

        worker._maybe_dual_patch_aitown_spark_meta(sess, correlation_id="c1", patch_spark_meta={})  # must not raise


class TestUpsertChatHistoryRowModelClsParam:
    """The refactor that made model_cls a parameter must not change a single
    byte of the default (ChatHistoryLogSQL) call shape -- test_chat_history_
    turn_coalesce.py already covers that exhaustively and stays green
    unmodified; this only covers the new non-default path directly."""

    def test_model_cls_targets_the_named_table(self):
        captured = {}

        class _Capturing:
            def execute(self, stmt):
                captured["stmt"] = stmt

        worker.upsert_chat_history_row(
            _Capturing(), {"id": "abc", "client_meta": {"x": 1}}, model_cls=worker.AitownChatHistoryLogSQL
        )

        sql = _compiled_sql(captured["stmt"])
        assert "aitown_chat_history_log" in sql

    def test_default_model_cls_is_still_chat_history_log(self):
        captured = {}

        class _Capturing:
            def execute(self, stmt):
                captured["stmt"] = stmt

        worker.upsert_chat_history_row(_Capturing(), {"id": "abc", "correlation_id": "abc"})

        sql = _compiled_sql(captured["stmt"])
        assert "insert into chat_history_log" in sql
