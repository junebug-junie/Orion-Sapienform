"""Tests for `response_identity` on chat_history_log.

Covers the actual bug: `_ensure_chat_history_from_message` (the message-path
merge into a turn's chat_history_log row) received `model`/`speaker` on
every call already -- room_claude_relay set `model`, everything set
`speaker` -- but silently dropped both before building `values`, so
`response_identity` never landed regardless of what a producer sent.

Same dynamic-import convention as test_aitown_chat_history_dual_write.py (no
real Postgres; DB access is faked at the session boundary).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_response_identity_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))
SPEC.loader.exec_module(worker)


class _NestedCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _CapturingSession:
    """Fake session standing in for both `_ensure_chat_history_from_message`'s
    own `session_factory()` session and `upsert_chat_history_row`'s target."""

    def __init__(self) -> None:
        self.executed: list[Any] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def execute(self, stmt):
        self.executed.append(stmt)

    def begin_nested(self):
        return _NestedCtx()

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def _compiled_values(stmt) -> dict[str, Any]:
    """pg_insert(...).values(**values) keeps the bind params reachable off
    the compiled statement's parameters, independent of the ON CONFLICT SET
    clause (which is what test_aitown_chat_history_dual_write.py's plain
    string-contains checks target instead)."""
    compiled = stmt.compile()
    return dict(compiled.params)


@pytest.fixture()
def fake_session_factory(monkeypatch: pytest.MonkeyPatch):
    sessions: list[_CapturingSession] = []

    def _factory():
        sess = _CapturingSession()
        sessions.append(sess)
        return sess

    monkeypatch.setattr(worker, "session_factory", _factory)
    return sessions


class TestResponseIdentityFromMessage:
    def test_assistant_message_with_model_sets_response_identity(self, fake_session_factory):
        worker._ensure_chat_history_from_message(
            correlation_id="corr-1",
            session_id="sess-1",
            role="assistant",
            content="hello there",
            memory_status=None,
            memory_tier=None,
            client_meta=None,
            model="claude-sonnet-5",
            speaker="Claude",
        )
        sess = fake_session_factory[0]
        assert sess.committed is True
        values = _compiled_values(sess.executed[0])
        assert values["response"] == "hello there"
        assert values["response_identity"] == "claude-sonnet-5"

    def test_assistant_message_falls_back_to_speaker_when_no_model(self, fake_session_factory):
        worker._ensure_chat_history_from_message(
            correlation_id="corr-2",
            session_id="sess-1",
            role="assistant",
            content="hi",
            memory_status=None,
            memory_tier=None,
            client_meta=None,
            model=None,
            speaker="Orion",
        )
        sess = fake_session_factory[0]
        values = _compiled_values(sess.executed[0])
        assert values["response_identity"] == "Orion"

    def test_assistant_message_omits_response_identity_when_neither_known(self, fake_session_factory):
        worker._ensure_chat_history_from_message(
            correlation_id="corr-3",
            session_id="sess-1",
            role="assistant",
            content="hi",
            memory_status=None,
            memory_tier=None,
            client_meta=None,
        )
        sess = fake_session_factory[0]
        values = _compiled_values(sess.executed[0])
        assert "response_identity" not in values

    def test_user_message_never_sets_response_identity(self, fake_session_factory):
        """Prompt-side rows must never get a response_identity, even if a
        caller mistakenly passed model/speaker alongside role='user'."""
        worker._ensure_chat_history_from_message(
            correlation_id="corr-4",
            session_id="sess-1",
            role="user",
            content="what's up",
            memory_status=None,
            memory_tier=None,
            client_meta=None,
            model="should-not-appear",
            speaker="Juniper",
        )
        sess = fake_session_factory[0]
        values = _compiled_values(sess.executed[0])
        assert values["prompt"] == "what's up"
        assert "response_identity" not in values
