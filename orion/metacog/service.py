from __future__ import annotations

"""orion/metacog/service.py

Causal-density scoring for MetacogEntryV1 (orion/schemas/metacog_entry.py).

Deliberately much smaller than orion/collapse/service.py: this model has no
self-report leg (no `numeric_sisters`), so there is nothing to blend against.
Score is derived purely from the real-artifact blend already sitting on
`MetacogRealState` (repair_pressure, substrate_eventfulness_score, a severity
read off turn_effect).

No store class here -- `apply_causal_density_to_entry` (the collapse-mirror
equivalent) is only ever called from services/orion-cortex-exec/app/executor.py
and services/orion-cortex-exec/app/collapse_verbs.py, both of which build a
CollapseMirrorEntryV2, not this model. Nothing else needs a standalone scoring
entry point for MetacogEntryV1 today, so cortex-exec calls this pure function
inline when building the entry instead of routing through a separate store --
a thinner seam than replicating CollapseMirrorStore for a single caller.
"""

import logging
from typing import Literal

from orion.schemas.metacog_entry import (
    MetacogCausalDensity,
    MetacogProvenance,
    MetacogRealState,
)

logger = logging.getLogger("orion.metacog.service")

# Thresholds against real orion-llm-gateway output
# (services/orion-llm-gateway/app/llm_uncertainty.py: mean_top1_margin is the
# top1-vs-top2 logprob gap, low_margin_token_count counts tokens below that
# gateway's own low_margin threshold). Starting defaults, same calibration
# caveat as the weights above -- not fit against real outcome data yet.
LOW_MARGIN_TOKEN_COUNT_DEGRADED = 1
LOW_MARGIN_TOKEN_COUNT_CRITICAL = 5
MEAN_TOP1_MARGIN_CRITICAL = 0.2

NON_OK_STEP_COUNT_DEGRADED = 1
NON_OK_STEP_COUNT_CRITICAL = 2

# Starting-default blend weights, not derived from any calibration data yet.
# Expect these to get retuned once we have real metacog-lane outcomes to check
# against -- same caveat as orion/collapse/service.py's own weights.
REPAIR_PRESSURE_WEIGHT = 0.5
SUBSTRATE_EVENTFULNESS_WEIGHT = 0.3
TURN_EFFECT_SEVERITY_WEIGHT = 0.2

IS_CAUSALLY_DENSE_THRESHOLD = 0.6


def turn_effect_severity(turn_effect: dict | None) -> float | None:
    """Read a severity scalar off turn_effect["turn"] (the delta block already
    computed by orion/schemas/telemetry/turn_effect.py), or None if absent.

    Severity = magnitude of the largest single delta component
    (valence/energy/coherence/novelty), each already clamped to [-1, 1] by
    the producer, normalized here to [0, 1].
    """
    if not isinstance(turn_effect, dict):
        return None
    turn = turn_effect.get("turn")
    if not isinstance(turn, dict):
        return None
    magnitudes = [
        abs(float(v))
        for v in turn.values()
        if isinstance(v, (int, float))
    ]
    if not magnitudes:
        return None
    return max(0.0, min(1.0, max(magnitudes)))


def _label_for_score(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= IS_CAUSALLY_DENSE_THRESHOLD:
        return "dense"
    if score >= 0.25:
        return "salient"
    return "ambient"


def compute_causal_density(state: MetacogRealState) -> MetacogCausalDensity:
    """Score causal_density purely from the real-artifact blend in `state`.

    No self-report leg exists in MetacogEntryV1, so there is nothing to blend
    the real-artifact evidence against -- unlike
    orion/collapse/service.py::apply_causal_density_to_entry, which blends a
    self-report score with phi/relational evidence.
    """
    components: list[tuple[float, float]] = []
    rationale_bits: list[str] = []

    repair_pressure = state.repair_pressure
    if repair_pressure is not None:
        repair_score = max(0.0, min(1.0, float(repair_pressure.level) * float(repair_pressure.confidence)))
        components.append((REPAIR_PRESSURE_WEIGHT, repair_score))
        rationale_bits.append(
            f"repair_pressure(level={repair_pressure.level:.2f},confidence={repair_pressure.confidence:.2f})"
        )

    if state.substrate_eventfulness_score is not None:
        eventfulness_score = max(0.0, min(1.0, float(state.substrate_eventfulness_score)))
        components.append((SUBSTRATE_EVENTFULNESS_WEIGHT, eventfulness_score))
        rationale_bits.append(f"substrate_eventfulness({eventfulness_score:.2f})")

    severity = turn_effect_severity(state.turn_effect)
    if severity is not None:
        components.append((TURN_EFFECT_SEVERITY_WEIGHT, severity))
        rationale_bits.append(f"turn_effect_severity({severity:.2f})")

    if not components:
        return MetacogCausalDensity(label="ambient", score=0.0, rationale="no_real_artifact_evidence")

    total_weight = sum(w for w, _ in components)
    score = max(0.0, min(1.0, sum(w * v for w, v in components) / total_weight))

    return MetacogCausalDensity(
        label=_label_for_score(score),
        score=score,
        rationale="computed_from_real_artifacts: " + ", ".join(rationale_bits),
    )


def compute_severity(
    *,
    llm_uncertainty: dict | None,
    non_ok_step_count: int,
) -> Literal["nominal", "degraded", "critical"]:
    """Discrete severity off real numbers already on the entry -- not a
    repeat of causal_density's continuous score. Two independent real
    signals: how many steps this turn didn't come back "ok" (from
    executor.py's own logs/merged_result), and how uncertain the LLM's own
    token generation was (orion-llm-gateway's real logprob-margin telemetry,
    not a self-rating)."""
    low_margin_count = 0
    mean_margin: float | None = None
    if isinstance(llm_uncertainty, dict) and llm_uncertainty.get("available"):
        low_margin_count = int(llm_uncertainty.get("low_margin_token_count") or 0)
        raw_margin = llm_uncertainty.get("mean_top1_margin")
        if isinstance(raw_margin, (int, float)):
            mean_margin = float(raw_margin)

    if (
        non_ok_step_count >= NON_OK_STEP_COUNT_CRITICAL
        or low_margin_count >= LOW_MARGIN_TOKEN_COUNT_CRITICAL
        or (mean_margin is not None and mean_margin < MEAN_TOP1_MARGIN_CRITICAL)
    ):
        return "critical"

    if (
        non_ok_step_count >= NON_OK_STEP_COUNT_DEGRADED
        or low_margin_count >= LOW_MARGIN_TOKEN_COUNT_DEGRADED
    ):
        return "degraded"

    return "nominal"


def compute_touches(state: MetacogRealState) -> list[str]:
    """Topology, not severity: which other real-artifact evidence this entry
    actually carries. Mechanically derived from which `state` fields are
    populated -- no new signal, just naming what's already there."""
    touches: list[str] = []
    if state.repair_pressure is not None:
        touches.append("relational")
    if state.substrate_eventfulness_score is not None:
        touches.append("substrate")
    if state.turn_effect is not None or state.turn_effect_evidence is not None:
        touches.append("affect")
    if state.biometrics is not None:
        touches.append("biometrics")
    if state.llm_uncertainty is not None and state.llm_uncertainty.get("available"):
        touches.append("generation")
    return touches


def compute_provenance(*, trigger_kind: str, touches: list[str]) -> MetacogProvenance:
    """Dynamic per-entry provenance, not a hardcoded constant. `source` names
    what actually fired this entry (the real trigger_kind, already on the
    entry); `impacts` reuses `touches` instead of a second, separate mapping
    -- the evidence that's present *is* what this entry is about."""
    impacts_map = {
        "relational": "relationship_thread",
        "substrate": "execution_trajectory",
        "affect": "field_channel:turn_effect",
        "biometrics": "field_channel:biometrics",
        "generation": "llm_generation_quality",
    }
    impacts = [impacts_map[t] for t in touches if t in impacts_map]
    return MetacogProvenance(
        source=f"cortex_exec.metacog_pipeline.{trigger_kind}",
        produces="metacog_entry",
        impacts=impacts,
    )
