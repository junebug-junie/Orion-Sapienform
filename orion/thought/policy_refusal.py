from __future__ import annotations

from dataclasses import dataclass

from orion.schemas.thought import ThoughtEventV1

TRUST_RUPTURE_DEFER_THRESHOLD = 0.65


@dataclass(frozen=True)
class DispositionDecision:
    disposition: str
    reasons: list[str]
    boundary_register: bool = False


def evaluate_thought_disposition(
    thought: ThoughtEventV1,
    *,
    association_stale: bool,
    coalition_ids: set[str],
) -> DispositionDecision:
    reasons: list[str] = []
    if not thought.imperative.strip():
        reasons.append("empty_imperative")
    if not thought.evidence_refs:
        reasons.append("missing_evidence_refs")
    elif not set(thought.evidence_refs).issubset(coalition_ids | set(thought.strain_refs)):
        reasons.append("evidence_refs_not_in_coalition")
    if association_stale and not thought.evidence_refs:
        reasons.append("stale_broadcast_no_evidence")
    trust = thought.trust_rupture_score
    if trust is not None and trust >= TRUST_RUPTURE_DEFER_THRESHOLD:
        return DispositionDecision("refuse", reasons + ["trust_rupture"], boundary_register=True)
    if reasons:
        return DispositionDecision("defer", reasons)
    if thought.disposition != "proceed":
        return DispositionDecision(thought.disposition, list(thought.disposition_reasons))
    return DispositionDecision("proceed", [])
