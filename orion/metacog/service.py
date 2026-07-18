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

from orion.schemas.metacog_entry import MetacogCausalDensity, MetacogRealState

logger = logging.getLogger("orion.metacog.service")

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
