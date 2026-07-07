from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.worker import EmbodimentWorker
from orion.schemas.embodiment import WorldPerceptionV1


def _worker() -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._speaking_conversations = set()
    w._opened_conversations = set()
    w._settings = SimpleNamespace(
        speech_enabled=True,
        speech_lane="quick",
        speech_verb="chat_quick",
        speech_timeout_sec=30.0,
        cortex_request_channel="orion:cortex:exec:request",
        cortex_result_prefix="orion:exec:result",
    )
    return w


def _perception_in_convo() -> WorldPerceptionV1:
    return WorldPerceptionV1(
        player_id="orion",
        position={"x": 0.0, "y": 0.0},
        nearby_players=[{"player_id": "p9", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "distance": 1.0}],
        active_conversation={
            "conversation_id": "conv1",
            "status": "participating",
            "participants": ["orion", "p9"],
            "messages": [{"author_id": "p9", "author": "Juniper", "text": "hey Orion"}],
        },
    )


def test_empty_reply_is_not_injected():
    w = _worker()
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="")), \
         patch("app.worker.aitown_client.send_input") as si, \
         patch("app.worker.aitown_client.convex_mutation") as cm:
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result is None
    si.assert_not_called()
    cm.assert_not_called()


def test_own_agent_only_no_speech_when_not_participant():
    w = _worker()
    w._orion_player_id = "someone_else"
    req = AsyncMock(return_value="hello")
    with patch.object(w, "_request_utterance", new=req), \
         patch("app.worker.aitown_client.send_input") as si:
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result is None
    req.assert_not_awaited()
    si.assert_not_called()


def test_injectable_reply_is_injected():
    w = _worker()
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hi Juniper!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}) as si, \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result == "Hi Juniper!"
    assert si.call_count >= 1


def test_finish_sending_message_uses_numeric_timestamp_not_uuid():
    """The 'void' regression: finishSendingMessage must carry a numeric `timestamp`
    (upstream schema) so the engine advances numMessages/lastMessage and the partner
    can perceive Orion's turn. `messageUuid` here gets the input dropped."""
    w = _worker()
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hi Juniper!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}) as si, \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        asyncio.run(w._speak_once(_perception_in_convo()))
    finish_calls = [
        c for c in si.call_args_list
        if c.kwargs.get("name") == "finishSendingMessage"
    ]
    assert finish_calls, "finishSendingMessage input was never sent"
    args = finish_calls[-1].kwargs["args"]
    assert "messageUuid" not in args, "finishSendingMessage must not carry messageUuid"
    assert isinstance(args.get("timestamp"), int) and args["timestamp"] > 0
    assert args["playerId"] == "orion"
    assert args["conversationId"] == "conv1"


def test_no_speech_when_orion_spoke_last():
    # Turn-taking: if the last message is Orion's, wait for a reply (no self-echo).
    w = _worker()
    perc = _perception_in_convo()
    perc.active_conversation["messages"] = [
        {"author_id": "p9", "author": "Juniper", "text": "hey"},
        {"author_id": "orion", "author": "Orion", "text": "hi there"},
    ]
    req = AsyncMock(return_value="again?")
    with patch.object(w, "_request_utterance", new=req), \
         patch("app.worker.aitown_client.send_input") as si:
        result = asyncio.run(w._speak_once(perc))
    assert result is None
    req.assert_not_awaited()
    si.assert_not_called()


def test_speaks_when_partner_spoke_last():
    w = _worker()
    perc = _perception_in_convo()
    perc.active_conversation["messages"] = [
        {"author_id": "orion", "author": "Orion", "text": "hi"},
        {"author_id": "p9", "author": "Juniper", "text": "how are you?"},
    ]
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Doing well!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(perc))
    assert result == "Doing well!"


def test_opens_empty_conversation_once():
    # Empty transcript -> Orion opens; a second empty tick must NOT re-open (spam guard).
    w = _worker()
    perc = _perception_in_convo()
    perc.active_conversation["messages"] = []
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hey there!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        first = asyncio.run(w._speak_once(perc))
        second = asyncio.run(w._speak_once(perc))
    assert first == "Hey there!"
    assert second is None
    assert "conv1" in w._opened_conversations


def test_no_speech_until_participating():
    w = _worker()
    perc = _perception_in_convo()
    perc.active_conversation["status"] = "walkingOver"
    req = AsyncMock(return_value="hello")
    with patch.object(w, "_request_utterance", new=req), \
         patch("app.worker.aitown_client.send_input") as si:
        result = asyncio.run(w._speak_once(perc))
    assert result is None
    req.assert_not_awaited()
    si.assert_not_called()


def test_write_message_uses_player_id_not_author():
    w = _worker()
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hi Juniper!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}) as cm:
        asyncio.run(w._speak_once(_perception_in_convo()))
    args = cm.call_args.args[1]
    assert args["playerId"] == "orion"
    assert "author" not in args


def test_speech_disabled_short_circuits():
    w = _worker()
    w._settings.speech_enabled = False
    req = AsyncMock(return_value="hello")
    with patch.object(w, "_request_utterance", new=req), \
         patch("app.worker.aitown_client.send_input") as si:
        result = asyncio.run(w._speak_once(_perception_in_convo()))
    assert result is None
    req.assert_not_awaited()
    si.assert_not_called()


def test_heartbeat_logs_once_then_throttles():
    """Observability seam: a healthy loop (all other logs exception-only) must still
    emit one INFO heartbeat, and it must throttle so a tight perception interval
    doesn't spam. This is the gap that forced live DB probing to diagnose the void bug."""
    w = _worker()
    w._orion_player_id = "orion"
    w._last_heartbeat_log_at = None
    perc = _perception_in_convo()
    with patch("app.worker.logger.info") as info:
        w._maybe_log_heartbeat(perc)
        first = info.call_count
        w._maybe_log_heartbeat(perc)  # immediate second call -> throttled
        second = info.call_count
    assert first == 1, "first heartbeat must log"
    assert second == 1, "second immediate heartbeat must be throttled"
    msg = info.call_args_list[0].args[0]
    assert "embodiment_heartbeat" in msg
