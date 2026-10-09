"""Sleep -> story: a completed sleep starts one narrative dream about what it replayed.

The narrative dream (cortex-orch `dream_cycle` verb -> `dreams` table) had no
scheduler; all 19 before 2026-10-09 were started by hand. Now it runs at the end
of every completed sleep, on the same tiredness gate, and gets that sleep's
replay as its primary material. Hypotheses are deliberately left out: they are a
blind experiment, offered to Orion later with the arm hidden.
"""
from __future__ import annotations

from orion.schemas.dream_cycle import DreamCycleV1
from orion.schemas.telemetry.dream import DreamInternalTriggerV1, DreamSleepDigestV1

REPLAY_TEXT_CHARS = 280
STORY_SOURCE = "orion-dream.sleep"


def sleep_digest(cycle: DreamCycleV1) -> DreamSleepDigestV1:
    replay = sorted(cycle.replay, key=lambda r: r.weight, reverse=True)
    return DreamSleepDigestV1(
        cycle_id=cycle.cycle_id,
        started_at=cycle.started_at,
        pressure=round(cycle.pressure.pressure, 2),
        overdue="overdue" in (cycle.note or ""),
        replay=[f"{r.source_kind}: {' '.join(r.text.split())[:REPLAY_TEXT_CHARS]}" for r in replay],
    )


def story_trigger(cycle: DreamCycleV1) -> DreamInternalTriggerV1 | None:
    """The trigger for this sleep's story, or None when the sleep did no work."""
    if cycle.status != "completed" or not cycle.replay:
        return None
    return DreamInternalTriggerV1(
        trigger_id=f"sleep:{cycle.cycle_id}",
        source=STORY_SOURCE,
        reason=f"end of sleep {cycle.cycle_id}",
        sleep=sleep_digest(cycle),
    )
