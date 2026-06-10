from __future__ import annotations

from orion.memory.crystallization.schemas import MemoryCrystallizationV1

KIND_BASE = {
    "stance": 0.85,
    "decision": 0.8,
    "procedure": 0.75,
    "contradiction": 0.7,
    "open_loop": 0.65,
    "attractor": 0.6,
    "failure_mode": 0.55,
    "semantic": 0.5,
    "episode": 0.45,
}

CONFIDENCE_BOOST = {
    "certain": 0.1,
    "likely": 0.05,
    "possible": 0.0,
    "uncertain": -0.05,
}


def score_salience(crystallization: MemoryCrystallizationV1) -> float:
    """Compute salience from kind, evidence strength, and confidence."""
    base = KIND_BASE.get(crystallization.kind, 0.5)
    evidence_boost = 0.0
    if crystallization.evidence:
        strengths = [ev.strength for ev in crystallization.evidence]
        evidence_boost = min(0.15, sum(strengths) / len(strengths) * 0.1)
    conf_boost = CONFIDENCE_BOOST.get(crystallization.confidence, 0.0)
    planning_boost = 0.05 if crystallization.planning_effects else 0.0
    score = base + evidence_boost + conf_boost + planning_boost
    return max(0.0, min(1.0, round(score, 3)))


def apply_salience(crystallization: MemoryCrystallizationV1) -> MemoryCrystallizationV1:
    updated = crystallization.model_copy(deep=True)
    updated.salience = score_salience(updated)
    return updated
