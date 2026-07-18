from __future__ import annotations

import logging
from typing import Any

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

logger = logging.getLogger("orion.equilibrium.repair_pressure_metacog_gate")


def build_repair_pressure_metacog_trigger(
    *,
    correlation_id: str,
    appraisal: dict[str, Any] | None,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    level_floor: float,
    confidence_floor: float,
) -> MetacogTriggerV1 | None:
    """Turn a live repair_pressure_v2 appraisal (published on
    orion:repair_pressure:appraisal by services/orion-hub/scripts/pre_turn_appraisal_wiring.py)
    into a "relational" metacog trigger.

    Replaces the retired orion/memory/turn_change_classify.py SHIFT gate
    (services/orion-equilibrium-service/app/relational_metacog_gate.py, deleted).
    `trigger_kind="relational"` is kept -- still the same conceptual trigger
    *category* in MetacogTriggerV1's docstring, just a different evidence
    source now (repair_pressure_v2 instead of turn_change_classify). See
    docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md.
    """
    if not isinstance(appraisal, dict):
        return None

    level = appraisal.get("level")
    confidence = appraisal.get("confidence")
    if not isinstance(level, (int, float)) or not isinstance(confidence, (int, float)):
        return None
    if float(level) < level_floor or float(confidence) < confidence_floor:
        return None

    evidence = appraisal.get("evidence") or []
    behavior_applied = appraisal.get("behavior_applied")
    reason = f"repair_pressure:level={float(level):.2f}:confidence={float(confidence):.2f}"

    return MetacogTriggerV1(
        trigger_kind="relational",
        reason=reason[:500],
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        signal_refs=[correlation_id] if correlation_id else [],
        upstream={
            "level": level,
            "level_label": appraisal.get("level_label"),
            "confidence": confidence,
            "evidence": evidence,
            "behavior_applied": behavior_applied,
        },
    )
