"""Memory episode boundary Fix 1: every persisted chat turn carries the wall clock.

orion-memory-consolidation reads ``turn.spark_meta["conversation_phase"]
["phase_change"]``. Before this, nothing wrote that key (0 of 3,586 window
turns carried it), so every window closed through the "unknown phase" branch.

Three Hub write paths are covered:
- the unified lane (Juniper's live default): stamp from the situation build;
- the legacy WS/HTTP turn publish (live today only for workflow commands):
  read-only stamp in ``publish_chat_turn``;
- Orion's unprompted outreach is NOT stamped: its rows never reach consolidation (empty prompt),
  so a stamp there had no reader (review 2026-10-02).
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import uuid
from typing import Any

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")

_CORR = str(uuid.UUID("21111111-2222-3333-4444-555555555555"))
_STAMP = {"phase_change": "resumed_thread", "delta_user_seconds": 6360, "crossed_day": False, "source": "situation_build"}


class _RecordingBus:
    enabled = True

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, channel: str, payload: Any) -> None:
        self.published.append((channel, payload))


def _run():
    from orion.schemas.harness_finalize import HarnessRunV1

    return HarnessRunV1(
        correlation_id=_CORR,
        final_text="an answer",
        finalize_ran=True,
        step_count=3,
        compliance_verdict="completed",
        grounding_status="grounded",
    )


@pytest.fixture()
def hub_settings(monkeypatch):
    from scripts.settings import settings

    monkeypatch.setattr(settings, "PUBLISH_CHAT_HISTORY_LOG", True)
    return settings


def _turn_payloads(bus: _RecordingBus) -> list[Any]:
    return [env.payload for _, env in bus.published if getattr(env, "kind", None) == "chat.history"]


def test_unified_turn_persists_the_situation_stamp(hub_settings, monkeypatch):
    import scripts.chat_history as chat_history
    from orion.hub.turn_orchestrator import _publish_unified_turn_chat_history

    async def _must_not_read(_sid):
        raise AssertionError("the unified lane already has a stamp; no second clock read")

    monkeypatch.setattr(chat_history, "read_conversation_phase_stamp_for_session", _must_not_read)
    bus = _RecordingBus()
    asyncio.run(
        _publish_unified_turn_chat_history(
            bus=bus,
            correlation_id=_CORR,
            session_id="sess-1",
            user_message="yup I'll be away from home",
            response_text="an answer",
            payload={"user_id": "juniper"},
            run=_run(),
            conversation_phase=dict(_STAMP),
        )
    )
    turns = _turn_payloads(bus)
    assert len(turns) == 1
    assert turns[0].spark_meta["conversation_phase"] == _STAMP


def test_turn_without_a_stamp_gets_a_read_only_one(hub_settings, monkeypatch):
    """Legacy-lane turns (workflow commands) and a unified turn whose situation
    build was disabled both reach publish_chat_turn with no stamp."""
    import scripts.chat_history as chat_history

    calls: list[str] = []

    async def _read(sid):
        calls.append(sid)
        return {"phase_change": "short_pause", "delta_user_seconds": 300, "crossed_day": False, "source": "session_state_read"}

    monkeypatch.setattr(chat_history, "read_conversation_phase_stamp_for_session", _read)
    env = chat_history.build_chat_turn_envelope(
        prompt="Run github compactor.",
        response="Workflow: GitHub Compactor",
        session_id="sess-2",
        correlation_id=_CORR,
        user_id="juniper",
        source_label="hub_ws",
        spark_meta={"mode": "brain"},
    )
    bus = _RecordingBus()
    asyncio.run(chat_history.publish_chat_turn(bus, env))
    turns = _turn_payloads(bus)
    assert calls == ["sess-2"]
    assert turns[0].spark_meta["conversation_phase"]["phase_change"] == "short_pause"
    assert turns[0].spark_meta["mode"] == "brain"


def test_unreadable_clock_publishes_the_turn_without_a_guess(hub_settings, monkeypatch):
    import scripts.chat_history as chat_history

    async def _none(_sid):
        return None

    monkeypatch.setattr(chat_history, "read_conversation_phase_stamp_for_session", _none)
    env = chat_history.build_chat_turn_envelope(
        prompt="hi", response="hello", session_id="s", correlation_id=_CORR, user_id=None
    )
    bus = _RecordingBus()
    asyncio.run(chat_history.publish_chat_turn(bus, env))
    turns = _turn_payloads(bus)
    assert len(turns) == 1
    assert "conversation_phase" not in (turns[0].spark_meta or {})


def _outreach():
    from scripts.endogenous_outreach import EndogenousOutreach

    return EndogenousOutreach.__new__(EndogenousOutreach)


def test_outreach_writes_no_conversation_phase_nobody_reads(monkeypatch):
    """Review 2026-10-02: outreach rows never reach consolidation (empty prompt), and consolidation
    reads spark_meta, not client_meta, so a stamp here was a write with no reader. Removed."""
    captured: dict = {}

    async def fake_publish(bus, envelopes):
        captured["env"] = envelopes[0]

    async def must_not_read(sid):
        raise AssertionError("outreach must not read the conversation clock")

    fake = types.ModuleType("scripts.chat_history")
    fake.publish_chat_history = fake_publish
    fake.build_chat_history_envelope = lambda **kw: types.SimpleNamespace(payload=kw)
    fake.read_conversation_phase_stamp_for_session = must_not_read
    monkeypatch.setitem(sys.modules, "scripts.chat_history", fake)
    outreach = _outreach()
    outreach._bus = object()
    asyncio.run(outreach._publish_history(text="hello", session_id="s", correlation_id="c", message_id="m"))
    assert "conversation_phase" not in captured["env"].payload["client_meta"]
