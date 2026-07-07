from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.worker import (
    JOURNAL_TRIGGER_CHANNEL,
    JOURNAL_TRIGGER_KIND,
    EmbodimentWorker,
)
from orion.embodiment.salience import SalienceState
from orion.journaler.schemas import JournalTriggerV1
from orion.schemas.embodiment import WorldPerceptionV1


def _worker(*, memory_enabled: bool = True) -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._salience = SalienceState()
    w._bus = SimpleNamespace()
    w._active_conversation_id = None
    w._active_conversation_partner = None
    w._active_conversation_utterances = 0
    w._settings = SimpleNamespace(
        memory_enabled=memory_enabled,
        service_name="orion-embodiment",
        service_version="0.1.0",
        node_name="athena",
    )
    return w


def _perception(**kwargs) -> WorldPerceptionV1:
    base = {"player_id": "orion", "position": {"x": 0.0, "y": 0.0}}
    base.update(kwargs)
    return WorldPerceptionV1(**base)


def test_completed_conversation_publishes_one_journal_trigger():
    w = _worker()
    with patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        ref = asyncio.run(
            w._maybe_journal_episode(
                {"type": "conversation_completed", "with": "Juniper", "utterances": 4}
            )
        )
    assert ref is not None
    assert pub.await_count == 1
    _bus, channel, env = pub.await_args.args[:3]
    assert channel == JOURNAL_TRIGGER_CHANNEL
    assert env.kind == JOURNAL_TRIGGER_KIND
    trigger = JournalTriggerV1.model_validate(env.payload)
    assert trigger.trigger_kind == "town_episode"
    assert trigger.source_kind == "embodiment"
    assert "Juniper" in trigger.summary


def test_proximity_publishes_no_journal_trigger():
    w = _worker()
    with patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        ref = asyncio.run(w._maybe_journal_episode({"type": "proximity", "player_id": "j"}))
    assert ref is None
    pub.assert_not_awaited()


def test_memory_disabled_short_circuits():
    w = _worker(memory_enabled=False)
    with patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        ref = asyncio.run(
            w._maybe_journal_episode(
                {"type": "conversation_completed", "with": "Juniper", "utterances": 4}
            )
        )
    assert ref is None
    pub.assert_not_awaited()


def test_perception_delta_journals_completed_conversation_once():
    """A conversation that ends across perception ticks journals exactly once,
    proving the gate is actually wired into the runtime perception loop."""
    w = _worker()
    convo = {"conversation_id": "c1", "with": "Juniper"}
    with patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        # Tick 1: conversation active, Orion contributes an utterance.
        asyncio.run(w._journal_from_perception(_perception(active_conversation=convo)))
        w._active_conversation_utterances += 1
        # Tick 2: conversation gone -> completion detected.
        asyncio.run(w._journal_from_perception(_perception(active_conversation=None)))
        completion_calls = [
            c for c in pub.await_args_list
            if JournalTriggerV1.model_validate(c.args[2].payload).source_ref == "c1"
        ]
    assert len(completion_calls) == 1
    trigger = JournalTriggerV1.model_validate(completion_calls[0].args[2].payload)
    assert trigger.trigger_kind == "town_episode"
    assert "Juniper" in trigger.summary


def test_perception_delta_encounter_dedupes():
    """First sighting of a player journals a first-encounter episode; repeat
    sightings across ticks are deduped by the salience state."""
    w = _worker()
    nearby = [{"player_id": "p1", "name": "Juniper", "position": {"x": 1.0, "y": 1.0}, "distance": 1.4}]
    with patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        asyncio.run(w._journal_from_perception(_perception(nearby_players=nearby)))
        asyncio.run(w._journal_from_perception(_perception(nearby_players=nearby)))
        encounter_calls = [
            c for c in pub.await_args_list
            if JournalTriggerV1.model_validate(c.args[2].payload).source_ref == "p1"
        ]
    assert len(encounter_calls) == 1
    trigger = JournalTriggerV1.model_validate(encounter_calls[0].args[2].payload)
    assert "Juniper" in trigger.summary


def test_perception_delta_silent_flyby_not_journaled():
    """A conversation with no Orion utterances is not a salient episode."""
    w = _worker()
    convo = {"conversation_id": "c1", "with": "Juniper"}
    with patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        asyncio.run(w._journal_from_perception(_perception(active_conversation=convo)))
        asyncio.run(w._journal_from_perception(_perception(active_conversation=None)))
        completion_calls = [
            c for c in pub.await_args_list
            if JournalTriggerV1.model_validate(c.args[2].payload).source_ref == "c1"
        ]
    assert completion_calls == []
