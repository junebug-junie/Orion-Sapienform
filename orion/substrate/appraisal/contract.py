"""Behavior consumer: repair_pressure signal → response contract mode.

This is intentionally a pure function over a contract dict. It does not
import or modify any existing chat pipeline. Callers can opt in by passing
their assembled contract through `apply_repair_pressure_contract`.
"""

from __future__ import annotations

import copy
from typing import Any

from orion.signals.models import OrionSignalV1

from .signal_bridge import SIGNAL_KIND


REPAIR_PRESSURE_DEBUG_KEY = "repair_pressure"
REPAIR_PRESSURE_CONTRACT_METADATA_KEY = "repair_pressure_contract"

_LEVEL_HIGH = 0.75
_LEVEL_MID = 0.45
_CONFIDENCE_MIN = 0.60

_RULES_REPAIR_CONCRETE: tuple[str, ...] = (
    "no broad architecture wandering",
    "no spiritual abstraction",
    "no unsolicited future roadmap unless asked",
    "answer with one concrete operational path",
    "include file/module boundaries",
    "include 'do not build' section",
    "include tests/acceptance checks",
    "acknowledge correction briefly",
    "prefer deterministic logic over LLM vibe scoring",
)

_RULES_CONCRETE_BIAS: tuple[str, ...] = (
    "be more specific",
    "show assumptions",
    "include next concrete action",
)


def _evidence_kinds_from_dimensions(dims: dict[str, float]) -> list[str]:
    candidate_kinds = (
        "specificity_demand",
        "trust_rupture",
        "coherence_gap",
        "repetition_failure",
        "operational_block",
        "explicit_repair_command",
        "assistant_accountability_demand",
    )
    return [k for k in candidate_kinds if dims.get(k, 0.0) > 0.0]


def apply_repair_pressure_contract(
    base_contract: dict[str, Any],
    signal: OrionSignalV1 | None,
) -> dict[str, Any]:
    """Return a new contract dict adjusted by the repair_pressure signal.

    Spec §11.1:
        level >= 0.75 and confidence >= 0.60  → mode = "repair_concrete"
        0.45 <= level < 0.75                  → mode = "concrete_bias"
        otherwise                              → unchanged
    """

    contract = copy.deepcopy(base_contract)

    if signal is None or signal.signal_kind != SIGNAL_KIND:
        return contract

    level = float(signal.dimensions.get("level", 0.0))
    confidence = float(signal.dimensions.get("confidence", 0.0))

    if level >= _LEVEL_HIGH and confidence >= _CONFIDENCE_MIN:
        mode_applied = "repair_concrete"
        contract["mode"] = mode_applied
        contract["rules"] = list(_RULES_REPAIR_CONCRETE)
    elif _LEVEL_MID <= level < _LEVEL_HIGH:
        mode_applied = "concrete_bias"
        contract["mode"] = mode_applied
        contract["rules"] = list(_RULES_CONCRETE_BIAS)
    else:
        return contract  # no change, no debug payload

    contract[REPAIR_PRESSURE_DEBUG_KEY] = {
        "level": level,
        "confidence": confidence,
        "mode_applied": mode_applied,
        "evidence_kinds": _evidence_kinds_from_dimensions(signal.dimensions),
        "causal_molecule_ids": list(signal.causal_parents),
        "source_event_id": signal.source_event_id,
    }
    return contract
