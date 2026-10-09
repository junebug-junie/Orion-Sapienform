"""Eval: the first real sleep under novelty pressure (dc-981127ddbddf, 2026-10-09
06:33 UTC) handed to its story dream. Real replay -> story_trigger -> the real
dream_cycle.j2, rendered the way cortex-exec renders it (jinja2, autoescape off).

Checks the story gets every replayed item (plus control-pair items, unlabeled),
the prompt stays small, and the payload survives cortex-orch's validation."""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "fixtures" / "sleep_cycle_dc-981127ddbddf_2026-10-09.json"
TEMPLATE = ROOT / "orion" / "cognition" / "prompts" / "dream_cycle.j2"
# The sleep section must not crowd the memory bundle: 12 replay + 2 control items x 280 chars max.
MAX_SLEEP_SECTION_CHARS = 5000


def _cycle():
    from orion.schemas.dream_cycle import DreamCycleV1

    d = json.loads(FIXTURE.read_text())
    return DreamCycleV1(cycle_id=d["cycle_id"], trigger="pressure", status=d["status"], started_at=d["started_at"],
                        ended_at=d["started_at"], pressure=d["pressure"], replay=d["replay"], note=d["note"])


def test_real_sleep_story_prompt_carries_every_replayed_item():
    from app.story import story_trigger
    from orion.schemas.telemetry.dream import DreamInternalTriggerV1

    cycle = _cycle()
    trigger = story_trigger(cycle, control_items=[cycle.replay[0].model_copy(update={
        "ref_id": "pool-only", "text": "a control-arm item from outside the replay"})])
    as_orch_sees_it = DreamInternalTriggerV1.model_validate(trigger.model_dump(mode="json")).model_dump(mode="json")
    prompt = Environment(autoescape=False).from_string(TEMPLATE.read_text()).render(
        memory_digest="(memories)", metadata={"dream_trigger": as_orch_sees_it})
    section = prompt[prompt.index("TONIGHT'S SLEEP"):prompt.index("MEMORY BUNDLE")]
    print(f"\nsleep section: {len(section)} chars, {len(trigger.sleep.material)} items\n{section}")
    assert len(trigger.sleep.material) == len(cycle.replay) + 1 == 13
    for r in cycle.replay:
        assert " ".join(r.text.split())[:60] in section
    assert "a control-arm item from outside the replay" in section
    assert "13.26 against a sleep line of 3.0" in section
    assert len(section) <= MAX_SLEEP_SECTION_CHARS
