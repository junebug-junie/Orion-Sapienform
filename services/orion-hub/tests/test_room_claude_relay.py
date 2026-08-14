"""Hub relay for the three-way room.

Focus is the two things that are easy to get wrong and invisible when wrong:
which socket a reply lands in, and whether a failed turn is audible.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from scripts.room_claude_relay import ROOM_CLAUDE_KIND, RoomClaudeRelay
from orion.schemas.room_claude import ExternalRoomResponderV1, RoomClaudeUtteranceV1


class FakeBus:
    enabled = True

    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, channel, envelope):
        payload = envelope.model_dump(mode="json") if hasattr(envelope, "model_dump") else envelope
        self.published.append((channel, payload))


def _relay(**kw) -> RoomClaudeRelay:
    relay = RoomClaudeRelay(
        request_channel="orion:room:claude:request",
        utterance_channel="orion:room:claude:utterance",
        enabled=True,
        **kw,
    )
    relay._bus = FakeBus()
    return relay


def _utterance(**kw) -> RoomClaudeUtteranceV1:
    base = dict(
        request_id="req-1",
        room_id="hub-direct",
        responder=ExternalRoomResponderV1(participant_id="claude", participant_name="Claude"),
        text="a real room reply",
        model="claude-sonnet-5",
        cost_usd=0.004,
        duration_ms=1200,
        ok=True,
    )
    base.update(kw)
    return RoomClaudeUtteranceV1(**base)


@pytest.mark.asyncio
async def test_invite_publishes_a_request_on_the_bus():
    relay = _relay()
    request = await relay.invite(
        prompt="what do you think?", invited_by="Juniper",
        session_id="sess-1", room_id="hub-direct",
    )
    channel, envelope = relay._bus.published[0]
    assert channel == "orion:room:claude:request"
    assert envelope["kind"] == "room.claude.request.v1"
    assert envelope["payload"]["prompt"] == "what do you think?"
    assert envelope["payload"]["invited_by"] == "Juniper"
    assert request.request_id in relay._pending


@pytest.mark.asyncio
async def test_malformed_transcript_entry_does_not_cost_the_invite():
    """The transcript is first-turn seeding, not the memory -- one bad history
    row must not stop Claude being invited."""
    relay = _relay()
    request = await relay.invite(
        prompt="hi", invited_by="Juniper", session_id="s", room_id="hub-direct",
        transcript=[
            {"speaker_id": "juniper", "speaker_name": "Juniper", "text": "good one"},
            {"garbage": True},
        ],
    )
    assert len(request.transcript) == 1
    assert relay._bus.published, "invite must still publish"


@pytest.mark.asyncio
async def test_reply_only_reaches_the_session_that_invited():
    """A room is a conversation, not a broadcast. Without session scoping every
    open tab would receive someone else's answer."""
    relay = _relay()
    mine: asyncio.Queue = asyncio.Queue()
    theirs: asyncio.Queue = asyncio.Queue()
    relay.register_connection("conn-mine", mine)
    relay.note_session("conn-mine", "sess-1")
    relay.register_connection("conn-theirs", theirs)
    relay.note_session("conn-theirs", "sess-2")

    await relay.invite(prompt="hi", invited_by="Juniper", session_id="sess-1", room_id="hub-direct")
    request_id = next(iter(relay._pending))

    await relay._handle_utterance({"payload": _utterance(request_id=request_id).model_dump(mode="json")})

    assert mine.qsize() == 1
    assert theirs.qsize() == 0
    frame = mine.get_nowait()
    assert frame["kind"] == ROOM_CLAUDE_KIND
    assert frame["speaker"] == "Claude"
    assert frame["llm_response"] == "a real room reply"
    assert frame["model"] == "claude-sonnet-5"
    assert frame["cost_usd"] == pytest.approx(0.004)


@pytest.mark.asyncio
async def test_failed_turn_is_surfaced_not_swallowed():
    """Silence is indistinguishable from Claude choosing not to speak, which is
    exactly what makes an outage invisible."""
    relay = _relay()
    q: asyncio.Queue = asyncio.Queue()
    relay.register_connection("c", q)
    relay.note_session("c", "sess-1")

    bad = _utterance(ok=False, text="", error="401 OAuth access token is invalid")
    await relay._handle_utterance({"payload": bad.model_dump(mode="json")})

    frame = q.get_nowait()
    assert frame["ok"] is False
    assert "401" in frame["llm_response"]


@pytest.mark.asyncio
async def test_history_is_published_with_the_responder_identity():
    """external_responder is the field this feature adds: every stored room
    turn until now assumed Orion answered."""
    relay = _relay()
    published: list = []

    async def _fake_publish(bus, envelopes):
        published.extend(envelopes)

    import scripts.chat_history as chat_history

    original = chat_history.publish_chat_history
    chat_history.publish_chat_history = _fake_publish
    try:
        await relay._handle_utterance({"payload": _utterance().model_dump(mode="json")})
    finally:
        chat_history.publish_chat_history = original

    assert published, "a successful room turn must be persisted"
    payload = published[0].payload if hasattr(published[0], "payload") else published[0]
    meta = payload.client_meta if hasattr(payload, "client_meta") else payload["client_meta"]
    assert meta["external_responder"]["participant_name"] == "Claude"
    assert meta["external_responder"]["participant_kind"] == "peer_ai"
    assert meta["room_claude"]["cost_usd"] == pytest.approx(0.004)


@pytest.mark.asyncio
async def test_failed_turn_is_not_persisted_as_a_room_turn():
    """An error banner is not something Claude said; storing it would put
    words in a participant's mouth."""
    relay = _relay()
    published: list = []

    async def _fake_publish(bus, envelopes):
        published.extend(envelopes)

    import scripts.chat_history as chat_history

    original = chat_history.publish_chat_history
    chat_history.publish_chat_history = _fake_publish
    try:
        await relay._handle_utterance(
            {"payload": _utterance(ok=False, text="", error="boom").model_dump(mode="json")}
        )
    finally:
        chat_history.publish_chat_history = original

    assert published == []


@pytest.mark.asyncio
async def test_bad_payload_is_ignored_without_raising():
    relay = _relay()
    await relay._handle_utterance({"payload": {"nonsense": True}})
    await relay._handle_utterance("not a dict")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_disabled_relay_starts_no_consumer_task():
    relay = RoomClaudeRelay(
        request_channel="a", utterance_channel="b", enabled=False,
    )
    await relay.start(FakeBus())
    assert relay._task is None
    await relay.stop()


@pytest.mark.asyncio
async def test_unregistered_connection_stops_receiving():
    relay = _relay()
    q: asyncio.Queue = asyncio.Queue()
    relay.register_connection("c", q)
    relay.note_session("c", "sess-1")
    relay.unregister_connection("c")
    await relay._handle_utterance({"payload": _utterance().model_dump(mode="json")})
    assert q.qsize() == 0
