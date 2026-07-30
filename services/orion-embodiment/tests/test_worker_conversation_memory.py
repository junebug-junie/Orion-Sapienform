from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.worker import CHAT_HISTORY_TURN_CHANNEL, SOCIAL_TURN_CHANNEL, SOCIAL_TURN_KIND, EmbodimentWorker
from orion.embodiment.salience import SalienceState
from orion.journaler.schemas import JournalTriggerV1
from orion.schemas.chat_history import ChatHistoryTurnV1
from orion.schemas.embodiment import WorldPerceptionV1
from orion.schemas.social_chat import SocialRoomTurnV1


def _worker(*, conversation_memory_enabled: bool = True, memory_enabled: bool = True) -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._bus = SimpleNamespace()
    w._speaking_conversations = set()
    w._opened_conversations = set()
    w._salience = SalienceState()
    w._active_conversation_id = None
    w._active_conversation_partner = None
    w._active_conversation_utterances = 0
    w._settings = SimpleNamespace(
        speech_enabled=True,
        speech_lane="quick",
        speech_verb="chat_quick",
        speech_timeout_sec=30.0,
        cortex_request_channel="orion:cortex:exec:request",
        cortex_result_prefix="orion:exec:result",
        speech_unified_enabled=False,
        hub_chat_url="http://orion-athena-hub:8080/api/chat",
        unified_timeout_sec=120.0,
        unified_session_prefix="aitown",
        memory_enabled=memory_enabled,
        conversation_memory_enabled=conversation_memory_enabled,
        service_name="orion-embodiment",
        service_version="0.1.0",
        node_name="athena",
    )
    return w


def _perception_in_convo(*, other_is_human: bool = False) -> WorldPerceptionV1:
    return WorldPerceptionV1(
        player_id="orion",
        position={"x": 0.0, "y": 0.0},
        nearby_players=[{"player_id": "p9", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "distance": 1.0}],
        active_conversation={
            "conversation_id": "conv1",
            "status": "participating",
            "participants": ["orion", "p9"],
            "other": {"player_id": "p9", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "is_human": other_is_human},
            "messages": [{"author_id": "p9", "author": "Juniper", "text": "hey Orion"}],
        },
    )


def _calls_on(pub, channel):
    return [c for c in pub.await_args_list if c.args[1] == channel]


def test_speak_once_publishes_to_both_channels_with_shared_correlation_id():
    w = _worker()
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hi Juniper!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        asyncio.run(w._speak_once(_perception_in_convo(other_is_human=True)))
    assert pub.await_count == 2
    chat_calls = _calls_on(pub, CHAT_HISTORY_TURN_CHANNEL)
    social_calls = _calls_on(pub, SOCIAL_TURN_CHANNEL)
    assert len(chat_calls) == 1 and len(social_calls) == 1

    chat_env = chat_calls[0].args[2]
    chat_turn = chat_env.payload if isinstance(chat_env.payload, ChatHistoryTurnV1) else ChatHistoryTurnV1.model_validate(chat_env.payload)
    social_env = social_calls[0].args[2]
    assert social_env.kind == SOCIAL_TURN_KIND
    social_turn = SocialRoomTurnV1.model_validate(social_env.payload)

    assert chat_turn.prompt == "hey Orion"
    assert chat_turn.response == "Hi Juniper!"
    assert social_turn.prompt == "hey Orion"
    assert social_turn.response == "Hi Juniper!"
    # Both publishes for one exchange share a correlation_id, so a later
    # journal entry can cross-reference the exact chat-log/social-turn rows.
    assert str(chat_turn.correlation_id) == str(social_turn.correlation_id)

    for turn in (chat_turn, social_turn):
        assert turn.client_meta["external_room"] == {"platform": "aitown", "room_id": "conv1"}
        assert turn.client_meta["external_participant"] == {
            "participant_id": "p9", "participant_name": "Juniper", "participant_kind": "human",
        }


def test_speak_once_tags_npc_participant_kind():
    w = _worker()
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hi Mira!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        asyncio.run(w._speak_once(_perception_in_convo(other_is_human=False)))
    social_turn = SocialRoomTurnV1.model_validate(_calls_on(pub, SOCIAL_TURN_CHANNEL)[0].args[2].payload)
    assert social_turn.client_meta["external_participant"]["participant_kind"] == "npc"


def test_speak_once_no_publish_when_conversation_memory_disabled():
    w = _worker(conversation_memory_enabled=False)
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hi Juniper!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        asyncio.run(w._speak_once(_perception_in_convo()))
    pub.assert_not_awaited()


def test_prompt_empty_when_orion_opens_conversation():
    w = _worker()
    perc = _perception_in_convo()
    perc.active_conversation["messages"] = []
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hey there!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        asyncio.run(w._speak_once(perc))
    social_turn = SocialRoomTurnV1.model_validate(_calls_on(pub, SOCIAL_TURN_CHANNEL)[0].args[2].payload)
    assert social_turn.prompt == ""
    assert social_turn.response == "Hey there!"


def test_publish_skipped_when_partner_id_unknown():
    w = _worker()
    perc = _perception_in_convo()
    perc.active_conversation["other"] = None
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hi!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        asyncio.run(w._speak_once(perc))
    pub.assert_not_awaited()


def _perception(**kwargs) -> WorldPerceptionV1:
    base = {"player_id": "orion", "position": {"x": 0.0, "y": 0.0}}
    base.update(kwargs)
    return WorldPerceptionV1(**base)


def test_journal_prompt_seed_carries_real_transcript_when_enabled():
    w = _worker(conversation_memory_enabled=True)
    convo = {"conversation_id": "c1", "with": "Juniper",
             "messages": [{"author": "Juniper", "text": "hey Orion"},
                          {"author": "Orion", "text": "hi there"}]}
    with patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        asyncio.run(w._journal_from_perception(_perception(active_conversation=convo)))
        w._active_conversation_utterances += 1
        asyncio.run(w._journal_from_perception(_perception(active_conversation=None)))
        completion_calls = [
            c for c in pub.await_args_list
            if JournalTriggerV1.model_validate(c.args[2].payload).source_ref == "c1"
        ]
    assert len(completion_calls) == 1
    trigger = JournalTriggerV1.model_validate(completion_calls[0].args[2].payload)
    assert trigger.prompt_seed == "Juniper: hey Orion\nOrion: hi there"


def test_journal_prompt_seed_absent_when_conversation_memory_disabled():
    w = _worker(conversation_memory_enabled=False)
    convo = {"conversation_id": "c1", "with": "Juniper",
             "messages": [{"author": "Juniper", "text": "hey Orion"}]}
    with patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        asyncio.run(w._journal_from_perception(_perception(active_conversation=convo)))
        w._active_conversation_utterances += 1
        asyncio.run(w._journal_from_perception(_perception(active_conversation=None)))
        completion_calls = [
            c for c in pub.await_args_list
            if JournalTriggerV1.model_validate(c.args[2].payload).source_ref == "c1"
        ]
    trigger = JournalTriggerV1.model_validate(completion_calls[0].args[2].payload)
    assert trigger.prompt_seed is None


def test_journal_spawned_correlation_id_links_to_last_exchange():
    w = _worker(conversation_memory_enabled=True)
    perc = _perception_in_convo(other_is_human=True)
    with patch.object(w, "_request_utterance", new=AsyncMock(return_value="Hi Juniper!")), \
         patch("app.worker.aitown_client.send_input", return_value={"ok": True}), \
         patch("app.worker.aitown_client.convex_mutation", return_value={"ok": True}), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        # Tick 1: perception loop sees the active conversation and tracks it.
        asyncio.run(w._journal_from_perception(perc))
        # Orion speaks -- publishes the exchange and records its correlation_id.
        asyncio.run(w._speak_once(perc))
        exchange_calls = _calls_on(pub, SOCIAL_TURN_CHANNEL)
        exchange_id = str(SocialRoomTurnV1.model_validate(exchange_calls[0].args[2].payload).correlation_id)
        # Tick 2: conversation ends -> journaled with spawned_correlation_id set.
        asyncio.run(w._journal_from_perception(_perception(active_conversation=None)))
        completion_calls = [
            c for c in pub.await_args_list
            if c.args[1] == "orion:actions:trigger:journal.v1"
            and JournalTriggerV1.model_validate(c.args[2].payload).source_ref == "conv1"
        ]
    assert len(completion_calls) == 1
    trigger = JournalTriggerV1.model_validate(completion_calls[0].args[2].payload)
    assert trigger.spawned_correlation_id == exchange_id
