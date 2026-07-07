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


def _worker(*, memory_enabled: bool = True) -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._salience = SalienceState()
    w._bus = SimpleNamespace()
    w._settings = SimpleNamespace(
        memory_enabled=memory_enabled,
        service_name="orion-embodiment",
        service_version="0.1.0",
        node_name="athena",
    )
    return w


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
