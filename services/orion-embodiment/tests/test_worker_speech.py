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
        # Unified town speech defaults to OFF in tests so the existing quick-path
        # cases exercise the legacy lane unchanged; unified cases opt in explicitly.
        speech_unified_enabled=False,
        hub_chat_url="http://orion-athena-hub:8080/api/chat",
        unified_timeout_sec=120.0,
        unified_session_prefix="aitown",
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


def test_worker_does_not_send_finish_sending_message_directly():
    """The 'void' regression, corrected: the worker must NOT send `finishSendingMessage`
    itself. `messages:writeMessage` already enqueues it server-side with a numeric
    timestamp. Sending a second one double-counted numMessages, and the pre-fix version
    (messageUuid, no timestamp) poisoned the shared engine — a malformed lastMessage
    that crashed every saveWorld/runStep and froze the whole town. writeMessage is the
    single source of the turn advance."""
    w = _worker()
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hi Juniper!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}) as si, \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}) as cm:
        asyncio.run(w._speak_once(_perception_in_convo()))
    finish_calls = [c for c in si.call_args_list if c.kwargs.get("name") == "finishSendingMessage"]
    assert not finish_calls, "worker must not send finishSendingMessage directly (writeMessage enqueues it)"
    # startTyping is still sent; writeMessage carries the message + turn advance.
    typing_calls = [c for c in si.call_args_list if c.kwargs.get("name") == "startTyping"]
    assert typing_calls, "startTyping should still be sent"
    assert cm.called and cm.call_args.args[0] == "messages:writeMessage"


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


def test_prompt_sent_to_cortex_centers_latest_partner_line():
    w = _worker()
    perc = _perception_in_convo()
    perc.active_conversation["messages"] = [
        {"author_id": "p9", "author": "Juniper", "text": "hey Orion"},
        {"author_id": "orion", "author": "Orion", "text": "I am with you."},
        {"author_id": "p9", "author": "Juniper", "text": "can you answer me?"},
    ]
    req = AsyncMock(return_value="Yes, I can answer you.")
    with patch.object(w, "_request_utterance", new=req), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}):
        result = asyncio.run(w._speak_once(perc))
    assert result == "Yes, I can answer you."
    prompt = req.await_args.args[0]
    assert "Your task is to answer this latest line from Juniper:\ncan you answer me?" in prompt
    assert "I am with you." in prompt


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
