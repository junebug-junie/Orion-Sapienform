"""Deterministic salience scoring for crystallization proposals.

Salience is a retrieval-priority prior, not truth. The governor records the
score; operators may override it before approval.
"""

from __future__ import annotations

from orion.schemas.memory_crystallization import MemoryCrystallizationV1

_KIND_BASE: dict[str, float] = {
    "stance": 0.85,
    "decision": 0.8,
    "contradiction": 0.75,
    "failure_mode": 0.7,
    "procedure": 0.65,
    "attractor": 0.6,
    "open_loop": 0.55,
    "semantic": 0.5,
    "episode": 0.4,
}

_CONFIDENCE_WEIGHT: dict[str, float] = {
    "certain": 1.0,
    "likely": 0.9,
    "possible": 0.7,
    "uncertain": 0.5,
}


def score_salience(proposal: MemoryCrystallizationV1) -> float:
    """Score salience in [0, 1] from kind, evidence mass, and confidence."""
    base = _KIND_BASE.get(proposal.kind, 0.5)

    evidence_mass = sum(ev.strength for ev in proposal.evidence)
    evidence_bonus = min(0.15, 0.03 * evidence_mass)

    link_bonus = min(0.05, 0.01 * len(proposal.links))

    score = (base + evidence_bonus + link_bonus) * _CONFIDENCE_WEIGHT.get(proposal.confidence, 0.7)
    return max(0.0, min(1.0, round(score, 4)))
