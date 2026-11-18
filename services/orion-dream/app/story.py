"""Sleep -> story: a completed sleep starts one narrative dream about what it worked on.

The narrative dream (cortex-orch `dream_cycle` verb -> `dreams` table) had no
scheduler; all 19 before 2026-10-09 were started by hand. Now it runs at the end
of every completed, saved sleep, on the same tiredness gate.

Blind experiment: the sleep's hypotheses are offered to Orion later with the arm
hidden. Dream-arm pairs are drawn from the replay, control pairs from the whole
pool. A story about the replay alone would make exactly the dream-arm items
familiar and bias the comparison. So the story gets every item from BOTH arms'
pairs plus the rest of the replay, in a seeded shuffle with no labels, weights or
order to tell them apart, and never the hypotheses themselves.
"""
from __future__ import annotations

import random
from typing import Sequence

from orion.schemas.dream_cycle import DreamCycleV1, ReplayItemV1
from orion.schemas.telemetry.dream import DreamInternalTriggerV1, DreamSleepDigestV1

ITEM_TEXT_CHARS = 280
STORY_SOURCE = "orion-dream.sleep"


def story_material(replay: Sequence[ReplayItemV1], control_items: Sequence[ReplayItemV1], seed: str) -> list[str]:
    items = {r.ref_id: r for r in replay}
    for r in control_items:
        items.setdefault(r.ref_id, r)
    lines = [f"{r.source_kind}: {' '.join(r.text.split())[:ITEM_TEXT_CHARS]}" for r in items.values()]
    random.Random(seed).shuffle(lines)
    return lines


def story_trigger(
    cycle: DreamCycleV1, *, control_items: Sequence[ReplayItemV1] = (), overdue: bool = False,
) -> DreamInternalTriggerV1 | None:
    """The trigger for this sleep's story, or None when the sleep did no work."""
    if cycle.status != "completed" or not cycle.replay:
        return None
    return DreamInternalTriggerV1(
        trigger_id=f"sleep:{cycle.cycle_id}",
        source=STORY_SOURCE,
        reason=f"end of sleep {cycle.cycle_id}",
        sleep=DreamSleepDigestV1(
            cycle_id=cycle.cycle_id,
            started_at=cycle.started_at,
            pressure=round(cycle.pressure.pressure, 2),
            threshold=cycle.pressure.threshold,
            overdue=overdue,
            material=story_material(cycle.replay, control_items, cycle.cycle_id),
        ),
    )


def story_envelope(trigger: DreamInternalTriggerV1, source):
    """The `dream.trigger` envelope cortex-orch turns into the one-shot story (dream_cycle verb).
    One builder for every publisher: the sleep's story path and the carry's zero-hop fallback."""
    from orion.core.bus.bus_schemas import BaseEnvelope

    return BaseEnvelope(kind="dream.trigger", source=source, payload=trigger.model_dump(mode="json"))
