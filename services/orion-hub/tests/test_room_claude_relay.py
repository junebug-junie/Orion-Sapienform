"""Hub relay for the three-way room.

Focus is the two things that are easy to get wrong and invisible when wrong:
which socket a reply lands in, and whether a failed turn is audible.
"""

from __future__ import annotations

import asyncio
import json
import pathlib

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
async def test_falsy_session_id_does_not_broadcast_to_every_open_tab():
    """A brand-new tab's first action can be clicking "Ask Claude" before it
    has a session_id yet. The old scoping guard --
    ``if session_id and entry.get("session_id") and entry[...] != session_id``
    -- short-circuited to False whenever EITHER side was falsy, so a null
    session_id reached every connection, identified or not. Nobody should
    receive Claude's reply when there is nothing to scope it by."""
    relay = _relay()
    a: asyncio.Queue = asyncio.Queue()
    b: asyncio.Queue = asyncio.Queue()
    relay.register_connection("conn-a", a)
    relay.note_session("conn-a", "sess-a")
    relay.register_connection("conn-b", b)
    # conn-b never calls note_session -- an unidentified socket, same as a
    # fresh tab that hasn't sent session_hello yet.

    await relay.invite(prompt="hi", invited_by="Juniper", session_id=None, room_id="hub-direct")
    request_id = next(iter(relay._pending))
    await relay._handle_utterance({"payload": _utterance(request_id=request_id, session_id=None).model_dump(mode="json")})

    assert a.qsize() == 0
    assert b.qsize() == 0


@pytest.mark.asyncio
async def test_connection_id_pins_the_reply_regardless_of_session_id():
    """When the caller has a connection_id (the manual invite route always
    passes one), it is authoritative -- a stale or mismatched noted session_id
    on the target socket must not matter."""
    relay = _relay()
    target: asyncio.Queue = asyncio.Queue()
    other: asyncio.Queue = asyncio.Queue()
    relay.register_connection("conn-target", target)
    relay.note_session("conn-target", "sess-stale")
    relay.register_connection("conn-other", other)
    relay.note_session("conn-other", "sess-current")

    await relay.invite(
        prompt="hi", invited_by="Juniper", session_id="sess-current", room_id="hub-direct",
        connection_id="conn-target",
    )
    request_id = next(iter(relay._pending))
    await relay._handle_utterance(
        {"payload": _utterance(request_id=request_id, session_id="sess-current").model_dump(mode="json")}
    )

    assert target.qsize() == 1
    assert other.qsize() == 0


@pytest.mark.asyncio
async def test_connection_id_falls_back_to_session_id_when_pinned_socket_is_gone():
    """A Claude turn takes real wall-clock time. If Juniper refreshes the tab
    while a reply is in flight, the pinned connection_id from invite() is now
    dead -- but the new socket typically kept the same session_id (persisted
    client-side, resent on reconnect). Falling back to session_id here is what
    keeps the reply from being silently dropped on an ordinary page refresh."""
    relay = _relay()
    reconnected: asyncio.Queue = asyncio.Queue()
    relay.register_connection("conn-new", reconnected)
    relay.note_session("conn-new", "sess-1")

    await relay.invite(
        prompt="hi", invited_by="Oríon", session_id="sess-1", room_id="hub-direct",
        trigger="auto", connection_id="conn-old-now-disconnected",
    )
    request_id = next(iter(relay._pending))
    await relay._handle_utterance(
        {"payload": _utterance(request_id=request_id, session_id="sess-1").model_dump(mode="json")}
    )

    assert reconnected.qsize() == 1, "must fall back to session_id, not drop the reply"


@pytest.mark.asyncio
async def test_stale_connection_id_with_no_session_match_drops_silently():
    """The fallback chain has an end: if the pinned connection is gone AND no
    live socket carries the session either, there is nothing left to deliver
    to. Must drop, not guess by broadcasting."""
    relay = _relay()
    unrelated: asyncio.Queue = asyncio.Queue()
    relay.register_connection("conn-unrelated", unrelated)
    relay.note_session("conn-unrelated", "sess-other")

    await relay.invite(
        prompt="hi", invited_by="Oríon", session_id="sess-1", room_id="hub-direct",
        trigger="auto", connection_id="conn-old-now-disconnected",
    )
    request_id = next(iter(relay._pending))
    await relay._handle_utterance(
        {"payload": _utterance(request_id=request_id, session_id="sess-1").model_dump(mode="json")}
    )

    assert unrelated.qsize() == 0


@pytest.mark.asyncio
async def test_failed_turn_is_surfaced_not_swallowed():
    """Silence is indistinguishable from Claude choosing not to speak, which is
    exactly what makes an outage invisible."""
    relay = _relay()
    q: asyncio.Queue = asyncio.Queue()
    relay.register_connection("c", q)
    relay.note_session("c", "sess-1")

    # session_id echoed back on the utterance, as the companion actually does
    # (it round-trips the request's session_id) -- without a session_id to
    # scope by, _push has nothing to route the failure to.
    bad = _utterance(session_id="sess-1", ok=False, text="", error="401 OAuth access token is invalid")
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


@pytest.mark.asyncio
async def test_a_pass_pushes_an_empty_frame_so_the_ui_unsticks():
    """Claude choosing silence is a real, billed turn. No bubble and no stored
    turn -- but a frame MUST still reach the socket.

    app.js clears the "thinking..." chip and re-enables the Ask Claude button
    only on a `room_claude_utterance` frame, so returning early on a pass left
    the chip spinning and the button dead until reload. Review finding.

    NOTE ON `session_id="sess-1"`: without it this test passed for the wrong
    reason. `_utterance()` sets no session_id, and with no session and no
    pending invite `_push`'s scoping chain drops the frame by design -- so the
    old `assert q.qsize() == 0` was satisfied by the frame never being
    addressed, not by the pass path declining to send one. Matching the
    registered connection is what makes this test exercise delivery at all.
    """
    relay = _relay()
    q: asyncio.Queue = asyncio.Queue()
    relay.register_connection("c", q)
    relay.note_session("c", "sess-1")

    published: list = []

    async def _fake_publish(bus, envelopes):
        published.extend(envelopes)

    import scripts.chat_history as chat_history

    original = chat_history.publish_chat_history
    chat_history.publish_chat_history = _fake_publish
    try:
        await relay._handle_utterance(
            {"payload": _utterance(
                text="", passed=True, cost_usd=0.0041, session_id="sess-1",
            ).model_dump(mode="json")}
        )
    finally:
        chat_history.publish_chat_history = original

    assert q.qsize() == 1, "a pass must still unstick the UI"
    frame = q.get_nowait()
    assert frame["kind"] == "room_claude_utterance", "the chip only clears on this kind"
    # Empty text is what suppresses the bubble on the client, before its own
    # `if (claudeText)` guard -- so no client change was needed.
    assert frame["llm_response"] == "", "a pass must not render a bubble"
    assert frame["passed"] is True
    assert frame["ok"] is True, "a pass is not a failure"
    assert published == [], "a pass must not be stored as something Claude said"


@pytest.mark.asyncio
async def test_a_pass_is_still_dropped_when_no_socket_matches():
    """The scoping chain still applies to a pass -- it is a normal frame, not
    a broadcast. Guards the inverse of the bug above: unsticking the UI must
    not mean pushing to every open tab."""
    relay = _relay()
    q: asyncio.Queue = asyncio.Queue()
    relay.register_connection("c", q)
    relay.note_session("c", "sess-1")

    await relay._handle_utterance(
        {"payload": _utterance(
            text="", passed=True, cost_usd=0.0041, session_id="someone-elses-session",
        ).model_dump(mode="json")}
    )
    assert q.qsize() == 0


def test_the_relay_has_no_auto_invite_surface_at_all():
    """The post-turn auto-invite is GONE, not merely disabled.

    Removed 2026-09-08: it shipped every private Hub chat turn to Claude and
    billed a call per turn. A config flag left off would have been one .env
    line away from coming back, so the rate gate, the flag and the
    `_last_auto_invite` bookkeeping are all deleted. This test fails if any of
    them is reintroduced without a deliberate decision.
    """
    relay = _relay()
    for gone in ("should_auto_invite", "should_fire_auto_invite",
                 "auto_respond", "auto_min_gap_sec", "_last_auto_invite"):
        assert not hasattr(relay, gone), f"{gone} is back -- see CLAUDE.md 0A, proposal mode"


def test_the_live_chat_path_does_not_invite_claude():
    """The caller side of the same kill.

    Asserted against the source rather than by driving a websocket turn: the
    handler is ~2,300 lines into a coroutine that needs a full turn's worth of
    fixtures, and what matters here is simply that no invite call survives in
    it. A grep-shaped test is honest about being a grep.
    """
    handler = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "websocket_handler.py"
    ).read_text()
    # BEHAVIOUR-SHAPED, NOT NAME-SHAPED. An earlier version of this test
    # asserted on `_room_relay.invite(` -- but this very file holds the relay
    # as `room_relay` (no underscore) elsewhere, and the handler uses that
    # spelling six times for its legitimate register/unregister calls. A
    # re-added invite written in that nearby style, or extracted into a
    # helper, would have passed every assertion while restoring exactly the
    # behaviour this test claims to prevent. Review finding.
    #
    # `.invite(` and `trigger="auto"` are each ZERO in the handler, and either
    # one is unavoidable for a real re-add: you cannot invite Claude without
    # calling invite, and you cannot get the pass licence without the trigger.
    assert ".invite(" not in handler, "something in the live chat path invites Claude again"
    assert 'trigger="auto"' not in handler
    # The old symbols too, so a straight revert is caught by name as well.
    assert "should_fire_auto_invite" not in handler
    assert "room_claude_auto_invite_failed" not in handler


@pytest.mark.asyncio
async def test_auto_trigger_is_marked_on_the_request():
    """The companion needs to know an invite was automatic, because that is
    what licenses Claude to stay quiet.

    Kept after the post-turn hook was removed: nothing produces `trigger="auto"`
    today, but the endogenous stuck-prior trigger will, and it needs the same
    pass licence. The contract outlives its first producer."""
    relay = _relay()
    auto = await relay.invite(
        prompt="Orion: something", invited_by="Orion", session_id="s",
        room_id="hub-direct", trigger="auto",
    )
    manual = await relay.invite(
        prompt="what do you think?", invited_by="Juniper", session_id="s",
        room_id="hub-direct",
    )
    assert auto.trigger == "auto"
    assert manual.trigger == "manual"
