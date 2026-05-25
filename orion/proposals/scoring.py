from __future__ import annotations

from orion.proposals.policy import ProposalPolicyV1, ProposalTemplateV1
from orion.schemas.self_state import SelfStateV1

PRESSURE_DIMENSIONS = frozenset({
    "execution_pressure",
    "resource_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "field_intensity",
    "uncertainty",
    "policy_pressure",
})


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def dimension_score(self_state: SelfStateV1, dimension_id: str) -> float:
    dim = self_state.dimensions.get(dimension_id)
    if dim is None:
        return 0.0
    return clamp01(dim.score)


def dimension_confidence(self_state: SelfStateV1, dimension_id: str) -> float:
    dim = self_state.dimensions.get(dimension_id)
    if dim is None:
        return self_state.overall_confidence
    return clamp01(dim.confidence)


def template_match_score(
    *,
    self_state: SelfStateV1,
    template: ProposalTemplateV1,
    policy: ProposalPolicyV1 | None = None,
) -> tuple[float, dict[str, float]]:
    contributions: dict[str, float] = {}
    for dim_id, weight in template.dimensions.items():
        policy_weight = 1.0
        if policy is not None:
            policy_weight = float(policy.dimension_weights.get(dim_id, 1.0))
        contributions[dim_id] = clamp01(
            dimension_score(self_state, dim_id) * float(weight) * abs(policy_weight)
        )
    match = max(contributions.values()) if contributions else 0.0
    return clamp01(match), contributions


def proposal_urgency(
    *,
    self_state: SelfStateV1,
    template: ProposalTemplateV1,
) -> float:
    scores = [
        dimension_score(self_state, dim_id)
        for dim_id in template.dimensions
        if dim_id in PRESSURE_DIMENSIONS or dim_id.endswith("_pressure")
    ]
    if not scores:
        scores = [dimension_score(self_state, d) for d in PRESSURE_DIMENSIONS]
    return clamp01(max(scores) if scores else self_state.overall_intensity)


def proposal_confidence(
    *,
    self_state: SelfStateV1,
    template: ProposalTemplateV1,
) -> float:
    confs = [
        dimension_confidence(self_state, dim_id)
        for dim_id in template.dimensions
    ]
    if not confs:
        return clamp01(self_state.overall_confidence)
    return clamp01(sum(confs) / len(confs))


def proposal_priority(
    *,
    base_priority: float,
    match_score: float,
    urgency: float,
    confidence: float,
) -> float:
    return clamp01(
        base_priority + 0.4 * match_score + 0.2 * urgency + 0.1 * confidence
    )


def proposal_risk(
    *,
    base_risk: float,
    self_state: SelfStateV1,
    template: ProposalTemplateV1,
) -> float:
    risk = float(base_risk)
    if template.kind in ("prepare_action", "request_policy_review"):
        risk += 0.10
    if template.required_policy_gate not in ("none", "read_only"):
        risk += 0.05
    if dimension_score(self_state, "reliability_pressure") >= 0.5:
        risk += 0.10
    if dimension_score(self_state, "uncertainty") >= 0.5:
        risk += 0.08
    if template.kind in ("observe", "inspect", "summarize") and template.required_policy_gate == "read_only":
        risk = min(risk, 0.15)
    return clamp01(risk)
