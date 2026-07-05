from __future__ import annotations

from typing import Any

from orion.schemas.thought import ThoughtEventV1

_RELATIONAL_TASK_MODES = frozenset({"reflective_dialogue", "playful_exchange"})
_RELATIONAL_FRAMES = frozenset({"reflective", "playful_relational"})


def _is_relational(thought: ThoughtEventV1) -> bool:
    sl = thought.stance_harness_slice
    return sl.task_mode in _RELATIONAL_TASK_MODES or sl.conversation_frame in _RELATIONAL_FRAMES


def _unique(items: list[str], *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def enforce_thought_stance_quality(
    thought: ThoughtEventV1,
    stance_inputs: dict[str, Any],
) -> tuple[ThoughtEventV1, bool]:
    """Harness profile of enforce_chat_stance_quality — no business-mode compressor on relational turns."""
    _ = stance_inputs
    changed = False
    enriched = thought.model_copy(deep=True)
    sl = enriched.stance_harness_slice

    if not _is_relational(enriched) and sl.task_mode == "triage":
        hazards = list(sl.response_hazards)
        for hazard in ("self_intro_on_operational_turn", "generic_sympathy_script"):
            if hazard not in hazards:
                hazards.append(hazard)
                changed = True
        sl.response_hazards = _unique(hazards, limit=8)
        priorities = list(sl.response_priorities)
        if "triage_operational_blockers_first" not in priorities:
            sl.response_priorities = _unique(
                ["triage_operational_blockers_first"] + priorities,
                limit=8,
            )
            changed = True

    if not enriched.imperative.strip():
        enriched.disposition = "defer"
        enriched.disposition_reasons = list(enriched.disposition_reasons) + ["empty_imperative"]
        changed = True

    if not enriched.evidence_refs:
        enriched.disposition = "defer"
        enriched.disposition_reasons = list(enriched.disposition_reasons) + ["missing_evidence_refs"]
        changed = True

    return enriched, changed
