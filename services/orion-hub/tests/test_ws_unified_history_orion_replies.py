"""Orion's own live replies must reach the next turn's RECENT CONVERSATION.

Bug (found 2026-10-16 by the PR #2725 agent): the unified Orion/Agent lane in
``websocket_endpoint`` appended Juniper's message to the per-connection
``history`` list, ran the turn, and ``continue``d -- it never appended Orion's
reply and never trimmed the list. So within one open socket, the second turn's
continuity held Juniper's two messages back to back and nothing Orion said in
between. The list also grew without bound for the life of the tab.

These tests drive the real ``websocket_endpoint`` with a fake socket (same
pattern as test_workflow_schedule_runtime_paths.py) and a stubbed
``run_unified_turn`` that records the continuity it was handed.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState

# Settings validation runs at import time; same defaults the other
# websocket_endpoint driver (test_workflow_schedule_runtime_paths.py) sets.
for _key, _val in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(_key, _val)

import scripts.websocket_handler as ws_handler  # noqa: E402
from scripts.chat_history_rehydrate import (  # noqa: E402
    append_live_assistant_reply,
    rehydrate_history,
    rows_to_history_messages,
    trim_history_window,
)
from orion.harness.cut_short import CUT_SHORT_MARKER  # noqa: E402


class _FakeWebSocket:
    def __init__(self, payloads: list[dict]) -> None:
        self._inbound = [json.dumps(p) for p in payloads]
        self.sent: list[Any] = []
        # The real enum: _safe_ws_send_json and the unified lane compare
        # against WebSocketState.CONNECTED, so a stand-in object would read as
        # "disconnected" and silently skip the code under test.
        self.client_state = WebSocketState.CONNECTED
        self.headers: dict = {}
        self.client = SimpleNamespace(host="127.0.0.1", port=9000)

    async def accept(self) -> None:
        return None

    async def receive_text(self) -> str:
        if self._inbound:
            return self._inbound.pop(0)
        self.client_state = WebSocketState.DISCONNECTED
        raise WebSocketDisconnect()

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def _final(text: str) -> list[dict]:
    return [{"type": "final", "correlation_id": "c", "llm_response": text}]


@pytest.fixture
def unified_ws(monkeypatch):
    """Wire websocket_endpoint so orion-mode turns hit a recording stub."""
    import scripts.main as hub_main
    import orion.hub.turn_orchestrator as orch

    for name, value in {
        "bus": None,
        "cortex_client": object(),
        "tts_client": None,
        "biometrics_cache": None,
        "notification_cache": None,
        "harness_step_relay": None,
        "rpc_bus": None,
        "presence_state": None,
        "endogenous_outreach": None,
        "room_claude_relay": None,
    }.items():
        monkeypatch.setattr(hub_main, name, value, raising=False)
    monkeypatch.setattr(ws_handler.settings, "ORION_UNIFIED_TURN_ENABLED", True, raising=False)
    monkeypatch.setattr(ws_handler.settings, "ORION_HARNESS_GOVERNOR_ENABLED", True, raising=False)
    monkeypatch.setattr(ws_handler.settings, "HUB_CONTEXT_TURNS", 10, raising=False)

    async def _no_sync(*_a, **_k):
        return None

    async def _passthrough(_ws, awaitable, **_k):
        return await awaitable

    monkeypatch.setattr(ws_handler, "_sync_fcc_sandbox_background", _no_sync)
    monkeypatch.setattr(ws_handler, "run_awaitable_cancel_on_ws_disconnect", _passthrough)
    monkeypatch.setattr(ws_handler.chat_turn_affect, "fire", lambda **_k: None)

    state: dict[str, Any] = {"calls": [], "replies": [], "history_lens": []}

    real_build = ws_handler.build_continuity_messages

    def _spy_build(*, history, **kwargs):
        # Length of the live per-connection list itself, not the capped copy
        # build_continuity_messages hands on -- that copy is bounded anyway, so
        # it cannot show whether the underlying list is.
        state["history_lens"].append(len(history))
        return real_build(history=history, **kwargs)

    monkeypatch.setattr(ws_handler, "build_continuity_messages", _spy_build)

    async def _fake_run_unified_turn(websocket, **kwargs):
        state["calls"].append(kwargs["continuity_messages"])
        return state["replies"].pop(0)

    monkeypatch.setattr(orch, "run_unified_turn", _fake_run_unified_turn)

    def _drive(payloads: list[dict], replies: list[list[dict]]) -> list[list[dict]]:
        state["calls"].clear()
        state["history_lens"].clear()
        state["replies"][:] = list(replies)
        asyncio.run(ws_handler.websocket_endpoint(_FakeWebSocket(payloads)))
        return list(state["calls"])

    _drive.state = state
    return _drive


def _turn(text: str, **extra) -> dict:
    return {"text_input": text, "mode": "orion", "session_id": "sid-hist", "no_write": True, **extra}


def test_second_turn_sees_orions_first_reply(unified_ws):
    calls = unified_ws(
        [_turn("hi orion"), _turn("what did you just say?")],
        [_final("Hello Juniper, the tide is out."), _final("I said the tide is out.")],
    )
    second = calls[1]
    roles = [(m["role"], m["content"]) for m in second if m["role"] != "system"]
    assert ("assistant", "Hello Juniper, the tide is out.") in roles
    # Order: Juniper, Orion, Juniper -- not two Juniper messages back to back.
    assert [r for r, _ in roles] == ["user", "assistant", "user"]
    # The ts shape #2725 established, so prefix.py renders the reply's time.
    reply = next(m for m in second if m["role"] == "assistant")
    assert isinstance(reply.get("ts"), str) and reply["ts"].endswith("+00:00")


def test_turn_without_a_final_frame_adds_no_assistant_text(unified_ws):
    """A turn_error partial_draft is never persisted to chat_history_log, so
    rehydrate never replays it; the live list must not invent it either.
    Juniper's own message from that turn deliberately stays (see
    append_live_assistant_reply's docstring), so the next turn sees two user
    messages in a row -- which is what actually happened."""
    calls = unified_ws(
        [_turn("first"), _turn("second")],
        [
            [{"type": "turn_error", "correlation_id": "c", "partial_draft": "I was in the middle of"}],
            _final("ok"),
        ],
    )
    assert all(m["role"] != "assistant" for m in calls[1])
    assert not any("middle of" in m["content"] for m in calls[1])


def test_cut_short_reply_keeps_its_marker(unified_ws):
    cut = f"{CUT_SHORT_MARKER}\nThis turn was stopped after 9 steps."
    calls = unified_ws([_turn("dig in"), _turn("and?")], [_final(cut), _final("ok")])
    reply = next(m for m in calls[1] if m["role"] == "assistant")
    assert reply["content"].startswith(CUT_SHORT_MARKER)


def test_history_stays_bounded_over_a_long_connection(unified_ws):
    turns = 3
    n = 12
    payloads = [_turn(f"msg {i}", context_turns=turns) for i in range(n)]
    calls = unified_ws(payloads, [_final(f"reply {i}") for i in range(n)])
    # The live list: system + `turns` pairs, plus this turn's new message.
    lens = unified_ws.state["history_lens"]
    assert len(lens) == n
    assert max(lens) <= 1 + 2 * turns + 1
    last = calls[-1]
    contents = [m["content"] for m in last]
    assert "reply 0" not in contents
    assert contents[-1] == f"msg {n - 1}"
    assert f"reply {n - 2}" in contents


# --------------------------------------------------------------- helpers


def test_append_skips_blank_and_error_shaped_text():
    h = [{"role": "system", "content": "s"}]
    assert append_live_assistant_reply(h, None) is False
    assert append_live_assistant_reply(h, "   ") is False
    assert append_live_assistant_reply(h, "[error] 400 from host") is False
    assert append_live_assistant_reply(h, "real words", ts="2026-10-16T00:00:00+00:00") is True
    assert h[-1] == {"role": "assistant", "content": "real words", "ts": "2026-10-16T00:00:00+00:00"}


def test_trim_keeps_system_and_newest_pairs():
    h = [{"role": "system", "content": "s"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(10)
    ]
    trim_history_window(h, 2)
    assert h[0]["role"] == "system"
    assert [m["content"] for m in h[1:]] == ["6", "7", "8", "9"]


@pytest.mark.parametrize("bad", [0, -3, "x", None])
def test_trim_with_degenerate_turns_never_duplicates_and_still_bounds(bad):
    """The legacy inline trim did ``history[:1] + history[-(2 * turns):]``;
    with turns=0 that slice is ``history[-0:]`` == the whole list, so it
    re-prepended the system message every turn instead of bounding."""
    h = [{"role": "system", "content": "s"}] + [{"role": "user", "content": str(i)} for i in range(5)]
    trim_history_window(h, bad)
    assert sum(1 for m in h if m["role"] == "system") == 1
    assert len(h) <= 3


def test_trim_without_system_message():
    h = [{"role": "user", "content": str(i)} for i in range(6)]
    trim_history_window(h, 1)
    assert [m["content"] for m in h] == ["4", "5"]


def test_live_list_is_never_duplicated_by_a_later_rehydrate(monkeypatch):
    """Rehydrate is restore-only: once the live list holds a turn, a second
    session_hello on the same socket must not splice persisted rows on top."""
    rows = [{"prompt": "hi", "response": "hello", "created_at": "2026-10-16T00:00:00+00:00"}]
    # Patch the globals rehydrate_history actually resolves against: conftest
    # drops cached scripts.* modules, so a fresh import could be a different
    # module object than the one this function was bound from.
    monkeypatch.setitem(rehydrate_history.__globals__, "fetch_recent_rows", lambda *a, **k: rows)
    h = [{"role": "system", "content": "s"}]
    assert asyncio.run(rehydrate_history(h, session_id="sid", max_turns=10, max_age_hours=48)) == 2
    append_live_assistant_reply(h, "a live reply")
    before = list(h)
    assert asyncio.run(rehydrate_history(h, session_id="sid", max_turns=10, max_age_hours=48)) == 0
    assert h == before


def test_live_reply_shape_matches_rehydrated_shape():
    """Same keys a rehydrated assistant message carries, so the harness
    renders live and restored replies identically."""
    restored = rows_to_history_messages(
        [{"prompt": "p", "response": "r", "created_at": "2026-10-16T00:00:00+00:00"}], max_turns=5
    )[-1]
    h: list[dict] = []
    append_live_assistant_reply(h, "r", ts="2026-10-16T00:00:00+00:00")
    assert set(h[-1]) == set(restored)


# --------------------------------------------------------------- legacy lane


class _RecordingCortexClient:
    """Plain brain-mode reply; records each request's continuity messages."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.messages: list[list[dict]] = []

    async def chat(self, req, correlation_id=None):
        from uuid import uuid4

        from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult

        self.messages.append([dict(m) if isinstance(m, dict) else m.model_dump() for m in (req.messages or [])])
        text = self.replies.pop(0)
        result = CortexClientResult(
            ok=True,
            mode="brain",
            verb="chat_general",
            status="success",
            final_text=text,
            memory_used=False,
            recall_debug={},
            steps=[],
            correlation_id=correlation_id or str(uuid4()),
            metadata={},
        )
        return CortexChatResult(cortex_result=result, final_text=text)


@pytest.mark.usefixtures("pre_turn_appraisal_off")
def test_legacy_lane_appends_reply_and_stays_bounded(monkeypatch):
    """The legacy (brain) lane now shares the same append + trim helpers;
    drive it through the real endpoint so a regression there is caught too."""
    import scripts.main as hub_main

    turns = 2
    n = 6
    client = _RecordingCortexClient([f"brain reply {i}" for i in range(n)])
    for name, value in {
        "bus": None,
        "cortex_client": client,
        "tts_client": None,
        "biometrics_cache": None,
        "notification_cache": None,
        "presence_state": None,
        "endogenous_outreach": None,
        "room_claude_relay": None,
    }.items():
        monkeypatch.setattr(hub_main, name, value, raising=False)

    async def _no_sync(*_a, **_k):
        return None

    monkeypatch.setattr(ws_handler, "_sync_fcc_sandbox_background", _no_sync)

    lens: list[int] = []
    real_build = ws_handler.build_continuity_messages

    def _spy_build(*, history, **kwargs):
        lens.append(len(history))
        return real_build(history=history, **kwargs)

    monkeypatch.setattr(ws_handler, "build_continuity_messages", _spy_build)

    payloads = [
        {"text_input": f"brain msg {i}", "mode": "brain", "session_id": "sid-legacy",
         "no_write": True, "context_turns": turns}
        for i in range(n)
    ]
    asyncio.run(ws_handler.websocket_endpoint(_FakeWebSocket(payloads)))

    assert len(client.messages) == n
    second = [m["content"] for m in client.messages[1]]
    assert "brain reply 0" in second
    assert max(lens) <= 1 + 2 * turns + 1
