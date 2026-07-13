from __future__ import annotations

from orion.memory.crystallization.schemas import CrystallizationConfidence, MemoryCrystallizationV1

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

# Threshold separating "weak/no real signal" evidence from "moderate" evidence in
# infer_confidence(). Mirrors the floor implied by the spec table's "avg < 0.3" cutoff.
_MODERATE_STRENGTH_FLOOR = 0.3


def infer_confidence(crystallization: MemoryCrystallizationV1) -> CrystallizationConfidence:
    """Deterministic confidence tier from evidence + recurrence. No LLM, no I/O.

    Reads `crystallization.evidence` (independent evidence sources at formation time)
    and `crystallization.dynamics.reinforcement_count` (recurrence — the same fact
    independently re-derived). Both are real evidentiary signals.

    Deliberately does NOT read `dynamics.last_recalled_at` or anything recall-related:
    being looked up is not evidence something is true, only that it's relevant. Callers
    must never invoke this from `recall_boost()` or any retrieval-time path — see
    `dynamics.py`'s module docstring / the reinforcement-decay wiring spec's hard
    invariant against conflating recall with reinforcement.

    Tiering (highest-support tier wins; evaluated strongest-to-weakest, not in the
    spec table's presentational uncertain->certain order, because the table's row
    conditions overlap and only a strength-ordered evaluation resolves that):

      1. Floor: no evidence, or evidence exists but is all weak on average
         (avg strength < 0.3), and nothing has independently recurred yet -> uncertain.
      2. Strong support: 3+ evidence sources, OR reinforcement_count >= 3, OR
         (2+ evidence sources AND reinforcement_count >= 1) -> certain.
      3. A single moderate-strength source with no recurrence yet is a first-cut
         guess, not yet corroborated -> possible.
      4. Everything else with real (if modest) support -- 1-2 evidence sources, or
         reinforcement_count 1-2 -> likely.
      5. Fallback (should be unreachable given 1-4, kept for safety) -> uncertain.
    """
    evidence = crystallization.evidence
    evidence_count = len(evidence)
    reinforcement_count = crystallization.dynamics.reinforcement_count
    avg_strength = (sum(ev.strength for ev in evidence) / evidence_count) if evidence_count else 0.0

    if (evidence_count == 0 or avg_strength < _MODERATE_STRENGTH_FLOOR) and reinforcement_count == 0:
        return "uncertain"

    if (
        evidence_count >= 3
        or reinforcement_count >= 3
        or (evidence_count >= 2 and reinforcement_count >= 1)
    ):
        return "certain"

    if evidence_count == 1 and reinforcement_count == 0 and avg_strength >= _MODERATE_STRENGTH_FLOOR:
        return "possible"

    if (1 <= evidence_count <= 2) or (1 <= reinforcement_count <= 2):
        return "likely"

    return "uncertain"


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
    """Compute confidence (from evidence/recurrence) and salience (which reads it).

    `infer_confidence()` runs first so `score_salience()`'s CONFIDENCE_BOOST lookup
    reads a real computed value on this same pass, not the schema's stale "likely"
    default. Both real formation call sites (proposer.py, intake_pipeline.py) go
    through this one function, so this is the single seam rather than duplicating
    the infer_confidence() call at each call site.
    """
    updated = crystallization.model_copy(deep=True)
    updated.confidence = infer_confidence(updated)
    updated.salience = score_salience(updated)
    return updated
