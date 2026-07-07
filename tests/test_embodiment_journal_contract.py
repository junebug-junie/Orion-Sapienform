from __future__ import annotations

from orion.journaler.schemas import JournalTriggerV1


def test_town_episode_trigger_kind_allowed():
    t = JournalTriggerV1(trigger_kind="town_episode", source_kind="embodiment",
                         summary="Talked with Juniper in the town")
    assert t.trigger_kind == "town_episode"
    assert t.source_kind == "embodiment"
