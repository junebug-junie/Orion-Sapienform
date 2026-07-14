from __future__ import annotations

from datetime import datetime, timezone

from orion.proposals.policy import ProposalPolicyV1, ProposalTemplateV1
from orion.proposals.scoring import (
    clamp01,
    dimension_score,
    proposal_confidence,
    proposal_priority,
    proposal_risk,
    proposal_urgency,
    template_match_score,
)
from orion.proposals.templates import (
    cast_policy_gate,
    cast_proposal_kind,
    cast_proposed_effect,
    cast_target_kind,
    template_title_description,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateV1


# The only recognized attention-binding literal in v1 -- no general
# binding-expression DSL, matched exactly against ProposalTemplateV1.target_binding.
ATTENTION_FIRST_TARGET_BINDING = "self_state.dominant_attention_targets[0]"

# Intersection of AttentionTargetSummaryV1.target_kind's Literal
# ("node", "capability", "channel", "edge", "field", "system") and
# ProposalCandidateV1.target_kind's Literal
# ("node", "capability", "field", "self_state", "service", "system").
# A resolved attention target outside this set fails closed to the
# template's literal target_id/target_kind rather than raising.
_ATTENTION_BOUND_TARGET_KINDS = frozenset({"node", "capability", "field", "system"})


def stable_proposal_frame_id(*, self_state_id: str, policy_id: str) -> str:
    return f"proposal.frame:{self_state_id}:{policy_id}"


def stable_proposal_id(*, template_key: str, self_state_id: str) -> str:
    return f"proposal:{template_key}:{self_state_id}"


def _resolve_binding_target(
    *,
    template: ProposalTemplateV1,
    self_state: SelfStateV1,
) -> tuple[str, str, str | None]:
    """Resolve a template's target_id/target_kind from live attention if the
    template declares a recognized binding and resolution succeeds; otherwise
    fail closed to the template's literal target_id/target_kind. Never raises.

    Returns (target_id, target_kind, binding_resolved_from).
    """
    if template.target_binding != ATTENTION_FIRST_TARGET_BINDING:
        return template.target_id, template.target_kind, None
    details = self_state.dominant_attention_target_details
    if not details:
        return template.target_id, template.target_kind, None
    resolved = details[0]
    if resolved.target_kind not in _ATTENTION_BOUND_TARGET_KINDS:
        return template.target_id, template.target_kind, None
    return resolved.target_id, resolved.target_kind, template.target_binding


def _build_candidate(
    *,
    template_key: str,
    template: ProposalTemplateV1,
    self_state: SelfStateV1,
    attention: FieldAttentionFrameV1 | None,
    policy: ProposalPolicyV1,
) -> ProposalCandidateV1:
    match_score, motivating_dimensions = template_match_score(
        self_state=self_state,
        template=template,
        policy=policy,
    )
    urgency = proposal_urgency(self_state=self_state, template=template)
    confidence = proposal_confidence(self_state=self_state, template=template)
    priority = proposal_priority(
        base_priority=template.base_priority,
        match_score=match_score,
        urgency=urgency,
        confidence=confidence,
    )
    risk = proposal_risk(
        base_risk=template.base_risk,
        self_state=self_state,
        template=template,
    )
    resolved_target_id, resolved_target_kind, binding_resolved_from = _resolve_binding_target(
        template=template,
        self_state=self_state,
    )
    title, description, reasons = template_title_description(
        template_key,
        target_id=resolved_target_id,
    )
    evidence_refs = [
        f"self_state:{self_state.self_state_id}",
        f"attention:{self_state.source_attention_frame_id}",
        f"field:{self_state.source_field_tick_id}",
    ]
    if attention is not None:
        evidence_refs.append(f"attention_frame:{attention.frame_id}")
    motivating_targets = list(self_state.dominant_attention_targets[:5])
    execution_intent: dict[str, str] = {
        "mode": "descriptive_only",
        "template": template_key,
        "policy_gate": template.required_policy_gate,
    }
    if template.kind == "request_policy_review":
        execution_intent["note"] = "policy_review_not_execution"
    return ProposalCandidateV1(
        proposal_id=stable_proposal_id(
            template_key=template_key,
            self_state_id=self_state.self_state_id,
        ),
        proposal_kind=cast_proposal_kind(template.kind),
        title=title,
        description=description,
        target_id=resolved_target_id,
        target_kind=cast_target_kind(resolved_target_kind),
        priority_score=priority,
        urgency_score=urgency,
        confidence_score=confidence,
        risk_score=risk,
        reversibility_score=clamp01(template.reversibility),
        motivating_dimensions=motivating_dimensions,
        motivating_targets=motivating_targets,
        evidence_refs=sorted(set(evidence_refs)),
        reasons=reasons,
        proposed_effect=cast_proposed_effect(template.proposed_effect),
        required_policy_gate=cast_policy_gate(template.required_policy_gate),
        execution_intent=execution_intent,
        binding_resolved_from=binding_resolved_from,
    )


def _overall_action_pressure(candidates: list[ProposalCandidateV1]) -> float:
    if not candidates:
        return 0.0
    return clamp01(max(c.priority_score for c in candidates))


def _overall_risk(candidates: list[ProposalCandidateV1]) -> float:
    if not candidates:
        return 0.0
    return clamp01(max(c.risk_score for c in candidates))


def _policy_required(
    *,
    candidates: list[ProposalCandidateV1],
    overall_risk: float,
    policy: ProposalPolicyV1,
) -> bool:
    if overall_risk >= policy.thresholds.policy_required_above_risk:
        return True
    return any(c.required_policy_gate not in ("none", "read_only") for c in candidates)


def _dominant_motivations(candidates: list[ProposalCandidateV1]) -> list[str]:
    counts: dict[str, float] = {}
    for candidate in candidates:
        for dim_id, weight in candidate.motivating_dimensions.items():
            counts[dim_id] = counts.get(dim_id, 0.0) + float(weight)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [dim_id for dim_id, _ in ranked[:5]]


def build_proposal_frame(
    *,
    self_state: SelfStateV1,
    attention: FieldAttentionFrameV1 | None,
    field: FieldStateV1 | None,
    policy: ProposalPolicyV1,
    previous_frame: ProposalFrameV1 | None = None,
    now: datetime | None = None,
    reverie_candidates: list[ProposalCandidateV1] | None = None,
) -> ProposalFrameV1:
    del previous_frame, field  # reserved for continuity in later revisions
    generated_at = now or datetime.now(timezone.utc)
    warnings: list[str] = []
    if attention is not None and attention.frame_id != self_state.source_attention_frame_id:
        warnings.append(
            f"attention_frame_mismatch:{attention.frame_id}!={self_state.source_attention_frame_id}"
        )

    built: list[ProposalCandidateV1] = []
    for template_key, template in policy.proposal_templates.items():
        built.append(
            _build_candidate(
                template_key=template_key,
                template=template,
                self_state=self_state,
                attention=attention,
                policy=policy,
            )
        )

    # Phase B: incorporate flag-gated spontaneous-thought proposals. These carry
    # source="reverie_thought" and an operator_review gate — they compete and are
    # policy-gated exactly like builder-native candidates, never auto-dispatched.
    for candidate in reverie_candidates or []:
        built.append(candidate)

    built.sort(key=lambda c: (-c.priority_score, c.proposal_id))
    active: list[ProposalCandidateV1] = []
    suppressed: list[ProposalCandidateV1] = []
    for candidate in built:
        if candidate.priority_score < policy.thresholds.suppress_below:
            suppressed.append(candidate)
        elif candidate.priority_score < policy.thresholds.min_priority:
            suppressed.append(candidate)
        else:
            active.append(candidate)

    active = active[: policy.limits.max_candidates]
    suppressed = suppressed[: policy.limits.max_suppressed]

    overall_risk = _overall_risk(active)
    return ProposalFrameV1(
        frame_id=stable_proposal_frame_id(
            self_state_id=self_state.self_state_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_self_state_id=self_state.self_state_id,
        source_self_state_generated_at=self_state.generated_at,
        source_attention_frame_id=self_state.source_attention_frame_id,
        source_field_tick_id=self_state.source_field_tick_id,
        proposal_policy_id=policy.policy_id,
        overall_action_pressure=_overall_action_pressure(active),
        overall_risk=overall_risk,
        policy_required=_policy_required(
            candidates=active,
            overall_risk=overall_risk,
            policy=policy,
        ),
        candidates=active,
        suppressed_candidates=suppressed,
        dominant_motivations=_dominant_motivations(active),
        warnings=warnings,
    )
