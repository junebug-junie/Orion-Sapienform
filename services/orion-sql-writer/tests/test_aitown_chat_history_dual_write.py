"""Tests for the AI Town chat-history table split
(docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md).

Covers ``_is_aitown_client_meta`` and ``_resolve_chat_history_model_cls`` in
``services/orion-sql-writer/app/worker.py`` -- the routing logic that
replaced Phase 1's additive dual-write (PR #1734) the same day it shipped,
once AI Town's own backend was confirmed dead and the dual-write bridge it
was built for became pure unneeded complexity. All DB access is faked at
the session boundary -- no real Postgres, matching the existing convention
in ``test_chat_history_turn_coalesce.py`` (still green, unmodified by this
change -- the ``model_cls`` parameter it exercises is untouched).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_aitown_routing_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))
SPEC.loader.exec_module(worker)


AITOWN_CLIENT_META = {"external_room": {"platform": "aitown", "room_id": "town-square"}}
HUB_CLIENT_META = {"external_room": {"platform": "hub"}}


class _FakeSession:
    """Fake session supporting the ``query(...).filter(...).with_for_update()
    .first()`` chain both ``_resolve_chat_history_model_cls`` and
    ``_apply_spark_meta_patch`` use. ``query_result_by_model`` lets a test
    control what each model class's query resolves to independently (e.g.
    "the primary table has this row, the mirror table doesn't").
    """

    def __init__(self, *, query_result_by_model: dict[type, Any] | None = None):
        self.executed: list[Any] = []
        self.lock_calls: list[Any] = []
        self.query_calls: list[type] = []
        self._query_result_by_model = query_result_by_model or {}
        self._current_model: type | None = None

    def execute(self, stmt, params=None):
        # params is not None => _lock_chat_history_row's advisory-lock call
        # (added to _apply_spark_meta_patch by the 2026-08-19
        # routing-awareness follow-up). Tracked separately so the existing
        # `sess.executed[0]` assertions below still see the real UPDATE.
        if params is not None:
            self.lock_calls.append((stmt, params))
            return None
        self.executed.append(stmt)

    def commit(self):
        pass

    def close(self):
        pass

    def query(self, entity):
        # _resolve_chat_history_model_cls queries a bare column
        # (AitownChatHistoryLogSQL.id), not the model class itself --
        # resolve back to the owning model either way.
        model_cls = getattr(entity, "class_", entity)
        self.query_calls.append(model_cls)
        self._current_model = model_cls
        return self

    def filter(self, *args, **kwargs):
        return self

    def with_for_update(self):
        return self

    def first(self):
        return self._query_result_by_model.get(self._current_model)


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


class TestLockChatHistoryRow:
    """Code review (2026-08-19) found _resolve_chat_history_model_cls's
    fallback lookup races across separate sessions/transactions --
    _lock_chat_history_row exists to close it. See that function's own
    docstring for the full scenario."""

    def test_issues_an_advisory_xact_lock_keyed_on_the_row_id(self):
        captured = {}

        class _Capturing:
            def execute(self, stmt, params=None):
                captured["stmt"] = stmt
                captured["params"] = params

        worker._lock_chat_history_row(_Capturing(), "row-123")

        sql = str(captured["stmt"]).lower()
        assert "pg_advisory_xact_lock" in sql
        assert "hashtext" in sql
        assert captured["params"] == {"row_id": "row-123"}


class TestResolveChatHistoryModelCls:
    def test_routes_to_primary_when_routing_disabled(self, monkeypatch: pytest.MonkeyPatch):
        """False is the real rollback path -- every row goes to
        chat_history_log regardless of client_meta, no exceptions."""
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", False)
        sess = _FakeSession()

        result = worker._resolve_chat_history_model_cls(sess, {"id": "x", "client_meta": AITOWN_CLIENT_META})

        assert result is worker.ChatHistoryLogSQL
        assert sess.query_calls == []  # no lookup needed, routing is off outright

    def test_routes_to_mirror_from_this_events_own_client_meta(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)
        sess = _FakeSession()

        result = worker._resolve_chat_history_model_cls(sess, {"id": "x", "client_meta": AITOWN_CLIENT_META})

        assert result is worker.AitownChatHistoryLogSQL
        assert sess.query_calls == []  # classified from this event alone, no lookup needed

    def test_routes_to_primary_for_non_aitown_client_meta(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)
        sess = _FakeSession()

        result = worker._resolve_chat_history_model_cls(sess, {"id": "x", "client_meta": HUB_CLIENT_META})

        assert result is worker.ChatHistoryLogSQL

    def test_falls_back_to_mirror_table_lookup_when_client_meta_absent(self, monkeypatch: pytest.MonkeyPatch):
        """The real bug this lookup exists to prevent: a message-path
        contribution with no client_meta of its own, arriving after the
        turn event already routed this id to the mirror table. Routing
        purely on this event's own (missing) signal would wrongly create a
        stray duplicate row in chat_history_log instead of merging into the
        row the turn event already created."""
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)
        sess = _FakeSession(query_result_by_model={worker.AitownChatHistoryLogSQL: ("existing-id",)})

        result = worker._resolve_chat_history_model_cls(sess, {"id": "x"})

        assert result is worker.AitownChatHistoryLogSQL
        assert sess.query_calls == [worker.AitownChatHistoryLogSQL]

    def test_falls_back_to_primary_when_no_id_and_no_client_meta(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)
        sess = _FakeSession()

        result = worker._resolve_chat_history_model_cls(sess, {})

        assert result is worker.ChatHistoryLogSQL
        assert sess.query_calls == []  # no id to look up

    def test_falls_back_to_primary_when_mirror_lookup_misses(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", True)
        sess = _FakeSession(query_result_by_model={worker.AitownChatHistoryLogSQL: None})

        result = worker._resolve_chat_history_model_cls(sess, {"id": "x"})

        assert result is worker.ChatHistoryLogSQL
        assert sess.query_calls == [worker.AitownChatHistoryLogSQL]


class TestUpsertChatHistoryRowModelClsParam:
    """The refactor that made model_cls a parameter must not change a single
    byte of the default (ChatHistoryLogSQL) call shape -- test_chat_history_
    turn_coalesce.py already covers that exhaustively and stays green
    unmodified; this only covers the non-default (routed) path directly."""

    def test_model_cls_targets_the_named_table(self):
        captured = {}

        class _Capturing:
            def execute(self, stmt):
                captured["stmt"] = stmt

        worker.upsert_chat_history_row(
            _Capturing(), {"id": "abc", "client_meta": {"x": 1}}, model_cls=worker.AitownChatHistoryLogSQL
        )

        sql = _compiled_sql(captured["stmt"])
        assert "insert into aitown_chat_history_log" in sql

    def test_default_model_cls_is_still_chat_history_log(self):
        captured = {}

        class _Capturing:
            def execute(self, stmt):
                captured["stmt"] = stmt

        worker.upsert_chat_history_row(_Capturing(), {"id": "abc", "correlation_id": "abc"})

        sql = _compiled_sql(captured["stmt"])
        assert "insert into chat_history_log" in sql


class TestApplySparkMetaPatchRouting:
    """``_apply_spark_meta_patch`` no longer assumes the row lives in
    chat_history_log -- routing may have sent it to the mirror table
    instead, and the patch payload carries no client_meta to reclassify
    from. Uses the real ``get_session``/session-factory plumbing
    (monkeypatched), so these go through ``ChatHistorySparkMetaPatchV1``
    validation like the real call path.
    """

    def _run(self, monkeypatch, *, sess, routing_enabled=True):
        monkeypatch.setattr(worker.settings, "sql_writer_aitown_routing_enabled", routing_enabled)
        monkeypatch.setattr(worker, "get_session", lambda: sess)
        monkeypatch.setattr(worker, "remove_session", lambda: None)
        return worker._apply_spark_meta_patch({"correlation_id": "c1", "spark_meta": {"new": True}})

    def test_patches_primary_table_when_row_found_there(self, monkeypatch: pytest.MonkeyPatch):
        class _Row:
            spark_meta = {"existing": True}

        sess = _FakeSession(query_result_by_model={worker.ChatHistoryLogSQL: _Row()})

        ok = self._run(monkeypatch, sess=sess)

        assert ok is True
        assert sess.query_calls == [worker.ChatHistoryLogSQL]  # found on first try, no mirror lookup
        sql = _compiled_sql(sess.executed[0])
        assert "update chat_history_log" in sql

    def test_falls_back_to_mirror_table_when_primary_misses(self, monkeypatch: pytest.MonkeyPatch):
        class _Row:
            spark_meta = {"existing": True}

        sess = _FakeSession(
            query_result_by_model={
                worker.ChatHistoryLogSQL: None,
                worker.AitownChatHistoryLogSQL: _Row(),
            }
        )

        ok = self._run(monkeypatch, sess=sess)

        assert ok is True
        assert sess.query_calls == [worker.ChatHistoryLogSQL, worker.AitownChatHistoryLogSQL]
        sql = _compiled_sql(sess.executed[0])
        assert "update aitown_chat_history_log" in sql

    def test_does_not_check_mirror_table_when_routing_disabled(self, monkeypatch: pytest.MonkeyPatch):
        sess = _FakeSession(query_result_by_model={worker.ChatHistoryLogSQL: None})

        ok = self._run(monkeypatch, sess=sess, routing_enabled=False)

        assert ok is False
        assert sess.query_calls == [worker.ChatHistoryLogSQL]  # no fallback attempted

    def test_returns_false_when_missing_from_both_tables(self, monkeypatch: pytest.MonkeyPatch):
        sess = _FakeSession(
            query_result_by_model={worker.ChatHistoryLogSQL: None, worker.AitownChatHistoryLogSQL: None}
        )

        ok = self._run(monkeypatch, sess=sess)

        assert ok is False
        assert sess.executed == []


class TestMirrorTableSchemaParity:
    """Code review (2026-08-19, PR #1734): no gate ensures
    AitownChatHistoryLogSQL stays column-for-column identical to
    ChatHistoryLogSQL. Without one, a future column added to
    chat_history_log alone would break routed writes for every AI-Town row
    silently. Kept unmodified by the dual-write -> routing change --
    exactly as true a concern for routing as it was for the additive
    mirror write. CLAUDE.md's "the right fix for [a] forgotten ... rule is
    not a louder prompt, it is a failing gate," applied to this exact
    failure mode.
    """

    def test_column_names_match_exactly(self):
        primary_cols = set(worker.ChatHistoryLogSQL.__table__.columns.keys())
        mirror_cols = set(worker.AitownChatHistoryLogSQL.__table__.columns.keys())
        assert mirror_cols == primary_cols, (
            f"aitown_chat_history_log has drifted from chat_history_log's columns -- "
            f"missing from mirror: {primary_cols - mirror_cols}, "
            f"extra on mirror: {mirror_cols - primary_cols}. Add the missing column(s) "
            f"to app/models/aitown_chat_history_log.py AND "
            f"services/orion-sql-db/manual_migration_aitown_chat_history_log_v1.sql "
            f"(or a follow-up migration) in the same change that adds them to "
            f"chat_history_log."
        )

    def test_column_types_match(self):
        primary_table = worker.ChatHistoryLogSQL.__table__
        mirror_table = worker.AitownChatHistoryLogSQL.__table__
        for name in primary_table.columns.keys():
            primary_type = type(primary_table.c[name].type)
            mirror_type = type(mirror_table.c[name].type)
            assert primary_type is mirror_type, (
                f"column {name!r}: chat_history_log is {primary_type.__name__}, "
                f"aitown_chat_history_log is {mirror_type.__name__}"
            )

    def test_primary_key_matches(self):
        assert [c.name for c in worker.ChatHistoryLogSQL.__table__.primary_key] == ["id"]
        assert [c.name for c in worker.AitownChatHistoryLogSQL.__table__.primary_key] == ["id"]
