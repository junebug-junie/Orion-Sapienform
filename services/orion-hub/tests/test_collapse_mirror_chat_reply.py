from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(SERVICE_DIR, "scripts")
for path in (SERVICE_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.codec import OrionCodec  # noqa: E402
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2  # noqa: E402
from orion.schemas.collapse_mirror_chat_reply import (  # noqa: E402
    COLLAPSE_MIRROR_CHAT_REPLY_KIND,
    CollapseMirrorChatReplyRequestV1,
)
from scripts.collapse_mirror_chat_reply import CollapseMirrorChatReplyHandler  # noqa: E402
from scripts.endogenous_outreach import EndogenousOutreach  # noqa: E402


class _FakeBus:
    def __init__(self) -> None:
        self.enabled = True
        self.published = []

    async def publish(self, channel: str, envelope) -> None:
        self.published.append((channel, envelope))

    def subscribe(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("tests call handle() directly")


class _ConsumeBus:
    """Fake bus that yields one codec-encoded pub/sub message then waits to stop."""

    def __init__(self, encoded_msgs: list) -> None:
        self.codec = OrionCodec()
        self.enabled = True
        self.published = []
        self._msgs = list(encoded_msgs)
        self._yielded = False

    async def publish(self, channel: str, envelope) -> None:
        self.published.append((channel, envelope))

    def subscribe(self, channel: str):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def iter_messages(self, pubsub):
        for msg in self._msgs:
            yield msg
        self._yielded = True
        while True:
            await asyncio.sleep(0.05)


def _always_fires():
    return None


def _make_outreach(**overrides) -> EndogenousOutreach:
    """Match test_endogenous_outreach._outreach required constructor kwargs."""
    base = dict(
        enabled=True,
        tick_interval_sec=300.0,
        min_cooldown_sec=0.0,
        daily_cap=4,
        quiet_start_hour=-1,
        quiet_end_hour=-1,
        timeout_sec=5.0,
        agent_lane_timeout_sec=5.0,
        notify_channel="orion:notify:in_app",
        fallback_session_id="MUST_NOT_USE",
        trigger_evaluator=_always_fires,
    )
    base.update(overrides)
    return EndogenousOutreach(**base)


def _entry(event_id: str = "evt-1") -> CollapseMirrorEntryV2:
    return CollapseMirrorEntryV2(
        event_id=event_id,
        observer="juniper",
        trigger="t",
        observer_state=["a"],
        type="reflect",
        emergent_entity="x",
        summary="mirror summary",
        mantra="m",
    )


def _request_env(event_id: str = "evt-1") -> BaseEnvelope:
    entry = _entry(event_id=event_id)
    req = CollapseMirrorChatReplyRequestV1(
        event_id=event_id,
        observer="juniper",
        mirror_text="### Collapse Mirror\n\n**Summary:** mirror summary\n",
        entry=entry,
    )
    return BaseEnvelope(
        kind=COLLAPSE_MIRROR_CHAT_REPLY_KIND,
        source=ServiceRef(name="orion-actions"),
        correlation_id=str(uuid4()),
        payload=req.model_dump(mode="json"),
    )


def _outreach_with_session(session_id: str = "live-sess") -> EndogenousOutreach:
    outreach = _make_outreach(fallback_session_id="MUST_NOT_USE")
    q: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", q, active_turn={})
    outreach.note_session("c1", session_id)
    return outreach


def _drain_queue(outreach: EndogenousOutreach) -> list:
    q = outreach._connections["c1"]["queue"]
    frames = []
    while not q.empty():
        frames.append(q.get_nowait())
    return frames


def _patch_history(monkeypatch):
    history = []

    async def _fake_publish_history(bus_arg, envelopes):
        history.extend(envelopes)

    monkeypatch.setattr(
        "scripts.chat_history.publish_chat_history",
        _fake_publish_history,
    )
    return history


@pytest.mark.asyncio
async def test_live_session_id_fail_closed_no_fallback() -> None:
    outreach = _make_outreach(fallback_session_id="fallback-sess")
    assert outreach.live_session_id() is None
    q: asyncio.Queue = asyncio.Queue()
    outreach.register_connection("c1", q, active_turn={})
    # connected but no session yet
    assert outreach.live_session_id() is None
    outreach.note_session("c1", "sess-9")
    assert outreach.live_session_id() == "sess-9"


@pytest.mark.asyncio
async def test_no_live_session_skips_without_you_or_generation(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = _make_outreach(fallback_session_id="fallback-sess")
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)
    called = {"turn": 0}

    async def _boom(*args, **kwargs):
        called["turn"] += 1
        raise AssertionError("must not generate without live session")

    import orion.hub.turn_orchestrator as turn_orchestrator

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _boom)
    result = await handler.handle(_request_env())
    assert result["status"] == "skipped"
    assert result["reason"] == "no_live_session"
    assert called["turn"] == 0
    # Optional held-back notify may land on the notify channel; never a You/history write.
    for channel, env in bus.published:
        assert channel == "orion:notify:in_app"
        assert getattr(env, "kind", None) == "notify.in_app.v1"
        payload = getattr(env, "payload", None) or {}
        if isinstance(payload, dict):
            assert payload.get("notification_type") == "collapse_mirror_held_back"
            assert "Collapse Mirror" not in str(payload.get("body_text") or "")


@pytest.mark.asyncio
async def test_live_session_injects_you_runs_chat_lane_and_delivers(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = _outreach_with_session("live-sess")
    outreach._bus = bus
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)
    turn_calls = []

    async def _fake_turn(**kwargs):
        turn_calls.append(kwargs)
        return [
            {
                "type": "final",
                "llm_response": "I hear that shift.",
                "fcc_model_label": None,
            }
        ]

    import orion.hub.turn_orchestrator as turn_orchestrator

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_turn)
    history = _patch_history(monkeypatch)

    result = await handler.handle(_request_env("evt-live"))
    assert result["status"] == "delivered"
    assert result["session_id"] == "live-sess"
    assert len(turn_calls) == 1
    payload = turn_calls[0]["payload"]
    assert payload.get("source") == "collapse_mirror_reply"
    assert "fcc_model_label" not in payload or payload.get("fcc_model_label") in (None, "")
    assert turn_calls[0]["user_message"].startswith("### Collapse Mirror")
    # You + Orion history (You first)
    roles = [getattr(e.payload, "role", None) for e in history]
    assert roles[0] == "user"
    assert "assistant" in roles
    assistant = next(e for e in history if getattr(e.payload, "role", None) == "assistant")
    tags = list(getattr(assistant.payload, "tags", None) or [])
    assert tags == ["collapse_mirror_reply"]
    assert "endogenous_outreach" not in tags
    meta = getattr(assistant.payload, "client_meta", None) or {}
    assert meta.get("unsolicited") is not True
    # socket got You then Orion
    frames = _drain_queue(outreach)
    assert any(f.get("kind") == "collapse_mirror_you" for f in frames)
    assert any(f.get("kind") == "orion_outreach" for f in frames)
    # notify must not look like unsolicited outreach
    notify = [env for ch, env in bus.published if ch == "orion:notify:in_app"]
    assert notify
    n_payload = getattr(notify[0], "payload", None) or {}
    assert n_payload.get("title") != "Orion reached out"
    assert n_payload.get("notification_type") == "collapse_mirror_reply"


@pytest.mark.asyncio
async def test_idempotent_on_event_id(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = _outreach_with_session("live-sess")
    outreach._bus = bus
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)
    turn_calls = {"n": 0}

    async def _fake_turn(**kwargs):
        turn_calls["n"] += 1
        return [{"type": "final", "llm_response": "ok", "fcc_model_label": None}]

    import orion.hub.turn_orchestrator as turn_orchestrator

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _fake_turn)
    monkeypatch.setattr(
        "scripts.chat_history.publish_chat_history",
        lambda *a, **k: asyncio.sleep(0),
    )

    env = _request_env("evt-dup")
    first = await handler.handle(env)
    second = await handler.handle(env)
    assert first["status"] == "delivered"
    assert second["status"] == "skipped"
    assert second["reason"] == "deduped"
    assert turn_calls["n"] == 1


@pytest.mark.asyncio
async def test_consume_loop_decodes_codec_encoded_message(monkeypatch) -> None:
    """Live entry path: codec.decode returns DecodeResult, not a bare envelope."""
    env = _request_env("evt-consume")
    codec = OrionCodec()
    bus = _ConsumeBus([{"data": codec.encode(env)}])
    outreach = _outreach_with_session("live-sess")
    outreach._bus = bus
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)
    handled: list = []

    async def _capture(incoming):
        handled.append(incoming)
        return {"status": "skipped", "reason": "test_capture", "event_id": "evt-consume"}

    monkeypatch.setattr(handler, "handle", _capture)

    task = asyncio.create_task(handler._consume_loop())
    try:
        for _ in range(100):
            if handled and bus._yielded:
                break
            await asyncio.sleep(0.02)
        assert len(handled) == 1
        got = handled[0]
        assert isinstance(got, BaseEnvelope)
        assert got.kind == COLLAPSE_MIRROR_CHAT_REPLY_KIND
        assert got.payload.get("event_id") == "evt-consume"
    finally:
        handler._stopping = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_context_overflow_leaves_you_without_orion(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = _outreach_with_session("live-sess")
    outreach._bus = bus
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)

    async def _overflow(**kwargs):
        return [
            {
                "type": "final",
                "llm_response": "partial overflow text",
                "context_overflow": True,
            }
        ]

    import orion.hub.turn_orchestrator as turn_orchestrator

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _overflow)
    history = _patch_history(monkeypatch)

    result = await handler.handle(_request_env("evt-overflow"))
    assert result["status"] == "failed"
    assert result.get("you_written") is True
    roles = [getattr(e.payload, "role", None) for e in history]
    assert "user" in roles
    assert "assistant" not in roles
    frames = _drain_queue(outreach)
    assert any(f.get("kind") == "collapse_mirror_you" for f in frames)
    assert not any(f.get("kind") == "orion_outreach" for f in frames)


@pytest.mark.asyncio
async def test_error_shaped_text_leaves_you_without_orion(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = _outreach_with_session("live-sess")
    outreach._bus = bus
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)

    async def _err_text(**kwargs):
        return [
            {
                "type": "final",
                # looks_like_error_text matches common upstream failure shapes
                "llm_response": "Error: context length exceeded",
            }
        ]

    import orion.hub.turn_orchestrator as turn_orchestrator

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _err_text)
    history = _patch_history(monkeypatch)

    result = await handler.handle(_request_env("evt-err-text"))
    assert result["status"] == "failed"
    assert result.get("you_written") is True
    roles = [getattr(e.payload, "role", None) for e in history]
    assert "user" in roles
    assert "assistant" not in roles
    frames = _drain_queue(outreach)
    assert any(f.get("kind") == "collapse_mirror_you" for f in frames)
    assert not any(f.get("kind") == "orion_outreach" for f in frames)


@pytest.mark.asyncio
async def test_turn_raise_leaves_you_without_orion(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = _outreach_with_session("live-sess")
    outreach._bus = bus
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)

    async def _boom(**kwargs):
        raise RuntimeError("turn exploded")

    import orion.hub.turn_orchestrator as turn_orchestrator

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _boom)
    history = _patch_history(monkeypatch)

    result = await handler.handle(_request_env("evt-raise"))
    assert result["status"] == "failed"
    assert result.get("you_written") is True
    assert "turn exploded" in str(result.get("reason") or "")
    roles = [getattr(e.payload, "role", None) for e in history]
    assert "user" in roles
    assert "assistant" not in roles
    frames = _drain_queue(outreach)
    assert any(f.get("kind") == "collapse_mirror_you" for f in frames)
    assert not any(f.get("kind") == "orion_outreach" for f in frames)


@pytest.mark.asyncio
async def test_empty_reply_leaves_you_without_orion(monkeypatch) -> None:
    bus = _FakeBus()
    outreach = _outreach_with_session("live-sess")
    outreach._bus = bus
    handler = CollapseMirrorChatReplyHandler(outreach=outreach, bus=bus)

    async def _empty(**kwargs):
        return [{"type": "final", "llm_response": "   "}]

    import orion.hub.turn_orchestrator as turn_orchestrator

    monkeypatch.setattr(turn_orchestrator, "execute_unified_turn", _empty)
    history = _patch_history(monkeypatch)

    result = await handler.handle(_request_env("evt-empty"))
    assert result["status"] == "failed"
    assert result.get("reason") == "empty_reply"
    assert result.get("you_written") is True
    roles = [getattr(e.payload, "role", None) for e in history]
    assert "user" in roles
    assert "assistant" not in roles
    frames = _drain_queue(outreach)
    assert any(f.get("kind") == "collapse_mirror_you" for f in frames)
    assert not any(f.get("kind") == "orion_outreach" for f in frames)
