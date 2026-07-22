from __future__ import annotations

from orion.proposals.policy import ProposalPolicyV1, ProposalTemplateV1

# 2026-07-22, SelfStateV1 burn: field_intensity and uncertainty removed. Both
# were composite, hand-tuned SelfStateV1 dimensions with no principled
# non-hand-tuned replacement (see orion/field/pressure.py's module docstring).
# The 4 remaining categories are real, direct channel-merge reads, unaffected
# by the burn.
PRESSURE_DIMENSIONS = frozenset({
    "execution_pressure",
    "resource_pressure",
    "reasoning_pressure",
    "reliability_pressure",
})


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def dimension_score(field_pressures: dict[str, float], dimension_id: str) -> float:
    return clamp01(field_pressures.get(dimension_id, 0.0))


def dimension_confidence(field_pressures: dict[str, float], dimension_id: str) -> float:
    """1.0 if this tick produced a real reading for this category, 0.0 if not.

    No principled per-dimension confidence formula survives the SelfStateV1
    burn (2026-07-22) -- self_state's old channel_dimension_confidence()
    (count + agreement, itself touched by the reverted equal-weighting
    reform) was not moved forward. This is an honest simplification: a
    binary presence flag, not a fabricated continuous score.
    """
    return 1.0 if dimension_id in field_pressures else 0.0


def template_match_score(
    *,
    field_pressures: dict[str, float],
    template: ProposalTemplateV1,
    policy: ProposalPolicyV1 | None = None,
) -> tuple[float, dict[str, float]]:
    contributions: dict[str, float] = {}
    for dim_id, weight in template.dimensions.items():
        policy_weight = 1.0
        if policy is not None:
            policy_weight = float(policy.dimension_weights.get(dim_id, 1.0))
        contributions[dim_id] = clamp01(
            dimension_score(field_pressures, dim_id) * float(weight) * abs(policy_weight)
        )
    match = max(contributions.values()) if contributions else 0.0
    return clamp01(match), contributions


def proposal_urgency(
    *,
    field_pressures: dict[str, float],
    template: ProposalTemplateV1,
) -> float:
    scores = [
        dimension_score(field_pressures, dim_id)
        for dim_id in template.dimensions
        if dim_id in PRESSURE_DIMENSIONS or dim_id.endswith("_pressure")
    ]
    if not scores:
        scores = [dimension_score(field_pressures, d) for d in PRESSURE_DIMENSIONS]
    # No SelfStateV1.overall_intensity fallback survives the burn -- honest
    # 0.0 ("no pressure data this tick") rather than a fabricated rollup.
    return clamp01(max(scores) if scores else 0.0)


def proposal_confidence(
    *,
    field_pressures: dict[str, float],
    template: ProposalTemplateV1,
) -> float:
    confs = [
        dimension_confidence(field_pressures, dim_id)
        for dim_id in template.dimensions
    ]
    if not confs:
        return 0.0
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
    field_pressures: dict[str, float],
    template: ProposalTemplateV1,
) -> float:
    risk = float(base_risk)
    if template.kind in ("prepare_action", "request_policy_review"):
        risk += 0.10
    if template.required_policy_gate not in ("none", "read_only"):
        risk += 0.05
    if dimension_score(field_pressures, "reliability_pressure") >= 0.5:
        risk += 0.10
    # The old "uncertainty" dimension risk bump is gone, not silently reading
    # 0.0 forever (2026-07-22 burn -- see PRESSURE_DIMENSIONS note above).
    if template.kind in ("observe", "inspect", "summarize") and template.required_policy_gate == "read_only":
        risk = min(risk, 0.15)
    return clamp01(risk)
