from __future__ import annotations

import logging
from typing import Any

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

logger = logging.getLogger("orion.equilibrium.relational_metacog_gate")

# orion/memory/turn_change_classify.py's SHIFT appraisal already distinguishes
# NONE/TOPIC/STANCE/REPAIR. REPAIR maps directly to the rupture-and-repair
# construct (Safran & Muran); TOPIC is a real subject change. STANCE is left
# out for now -- see docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md
# "Still open" section, not decided.
RELATIONAL_SHIFT_KINDS = ("REPAIR", "TOPIC")


def build_relational_metacog_trigger(
    *,
    correlation_id: str,
    turn_change_appraisal: dict[str, Any] | None,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    confidence_floor: float,
) -> MetacogTriggerV1 | None:
    """Turn a live turn_change_classify SHIFT appraisal into a relational metacog trigger.

    Reuses orion/memory/turn_change_classify.py's already-theory-anchored,
    already-scored SHIFT appraisal (published on
    orion:chat:history:spark_meta:patch by orion-memory-consolidation) instead
    of building a new topic-shift/rupture-repair detector.
    """
    if not isinstance(turn_change_appraisal, dict):
        return None
    if turn_change_appraisal.get("turn_change_status") != "ok":
        return None

    shift_kind = turn_change_appraisal.get("shift_kind")
    if shift_kind not in RELATIONAL_SHIFT_KINDS:
        return None

    confidence = turn_change_appraisal.get("confidence")
    if not isinstance(confidence, (int, float)) or float(confidence) < confidence_floor:
        return None

    novelty = turn_change_appraisal.get("novelty_score")
    reason = f"relational_shift:{str(shift_kind).lower()}:confidence={float(confidence):.2f}"

    return MetacogTriggerV1(
        trigger_kind="relational",
        reason=reason[:500],
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        signal_refs=[correlation_id] if correlation_id else [],
        upstream={
            "shift_kind": shift_kind,
            "novelty_score": novelty,
            "confidence": confidence,
            "turn_correlation_id": correlation_id,
        },
    )
