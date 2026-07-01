from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

SOCIAL_ROOM_PATH = HUB_ROOT / "scripts" / "social_room.py"
WS_PATH = HUB_ROOT / "scripts" / "websocket_handler.py"
SPEC = importlib.util.spec_from_file_location("hub_social_room", SOCIAL_ROOM_PATH)
assert SPEC and SPEC.loader
hub_social_room = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub_social_room)


def test_hub_direct_identity_uses_hub_platform_not_callsyne() -> None:
    identity = hub_social_room.hub_direct_room_identity("juniper")
    assert identity["external_room"]["platform"] == "hub"
    assert identity["external_room"]["room_id"] == "hub-direct"
    assert "callsyne" not in identity["external_room"]["platform"]


def test_social_room_client_meta_stamps_hub_direct_mode() -> None:
    meta = hub_social_room.social_room_client_meta(
        payload={"chat_profile": "social_room", "social_room_mode": "hub_direct"},
        route_debug={},
        trace_verb=None,
        memory_digest=None,
    )
    assert meta["social_room_mode"] == "hub_direct"


def test_websocket_handler_source_has_no_bridge_or_callsyne_wiring() -> None:
    source = WS_PATH.read_text(encoding="utf-8").lower()
    forbidden = (
        "orion-social-room-bridge",
        "callsyne_client",
        "post_message",
        "webhooks/callsyne",
        "external.room.post",
    )
    for needle in forbidden:
        assert needle not in source, f"websocket_handler must not reference {needle!r}"


@pytest.mark.asyncio
async def test_apply_hub_direct_social_room_mode_only_calls_social_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws_spec = importlib.util.spec_from_file_location("hub_ws_handler", WS_PATH)
    assert ws_spec and ws_spec.loader
    ws_module = importlib.util.module_from_spec(ws_spec)
    ws_spec.loader.exec_module(ws_module)

    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_fetch(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((path, params))
        return {"participant": {"participant_id": "juniper"}, "room": {"room_id": "hub-direct"}}

    api_routes = importlib.import_module("scripts.api_routes")
    monkeypatch.setattr(api_routes, "_fetch_social_memory", fake_fetch)

    result = await ws_module._apply_hub_direct_social_room_mode({"social_room_mode": "hub_direct"})
    assert result["social_room_mode"] == "hub_direct"
    assert result["chat_profile"] == "social_room"
    assert result["external_room"]["platform"] == "hub"
    assert calls == [
        (
            "/summary",
            {"platform": "hub", "room_id": "hub-direct", "participant_id": "juniper"},
        )
    ]


@pytest.mark.asyncio
async def test_apply_hub_direct_social_room_mode_noop_when_toggle_off() -> None:
    ws_spec = importlib.util.spec_from_file_location("hub_ws_handler", WS_PATH)
    assert ws_spec and ws_spec.loader
    ws_module = importlib.util.module_from_spec(ws_spec)
    ws_spec.loader.exec_module(ws_module)

    payload = {"mode": "brain", "text_input": "hello"}
    assert await ws_module._apply_hub_direct_social_room_mode(payload) == payload
