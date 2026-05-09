from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
from scripts import chat_history


class _FakeBus:
    enabled = True

    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel: str, envelope: object) -> None:
        self.published.append((channel, envelope))


def test_publish_social_room_turn_serializes_payload_dict(monkeypatch) -> None:
    bus = _FakeBus()
    monkeypatch.setattr(chat_history.settings, "PUBLISH_CHAT_HISTORY_LOG", True)

    turn = asyncio.run(
        chat_history.publish_social_room_turn(
            bus,
            prompt="hello",
            response="hi there",
            session_id="s-1",
            correlation_id="11111111-2222-3333-4444-555555555555",
            user_id="u-1",
            source_label="test",
            recall_profile="social.room.v1",
            trace_verb="chat_social_room",
            client_meta={"chat_profile": "social_room"},
            memory_digest="m1",
        )
    )

    assert turn is not None
    assert len(bus.published) == 1
    channel, env = bus.published[0]
    assert channel == chat_history.SOCIAL_ROOM_TURN_CHANNEL
    assert isinstance(env.payload, dict)
    assert env.payload.get("profile") == "social_room"
    assert env.payload.get("prompt") == "hello"
