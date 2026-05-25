from __future__ import annotations

from typing import Literal

from orion.policy.policy import SubstratePolicyV1
from orion.policy.rules import (
    hard_block_hits,
    is_read_only_candidate,
    kind_rule,
    stable_policy_decision_id,
)
from orion.proposals.scoring import clamp01
from orion.schemas.policy_decision_frame import PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateV1

DecisionLiteral = Literal[
    "approved_for_execution",
    "approved_read_only",
    "requires_operator_review",
    "deferred",
    "rejected",
]


def _policy_gate_for_decision(
    decision: DecisionLiteral,
    candidate_gate: str,
) -> str:
    if decision == "approved_read_only":
        return "read_only"
    if decision == "requires_operator_review":
        return "operator_review"
    if decision == "approved_for_execution":
        return "execution_policy"
    if decision in ("rejected", "deferred"):
        return "execution_policy" if candidate_gate == "execution_policy" else "autonomy_policy"
    return "none"


def _finish(
    *,
    candidate: ProposalCandidateV1,
    policy: SubstratePolicyV1,
    decision: DecisionLiteral,
    policy_gate: str | None,
    autonomy_tier: str,
    allowed_scope: str,
    risk_score: float,
    reversibility_score: float,
    confidence_score: float,
    reasons: list[str],
    evidence_refs: list[str],
    blocked_by: list[str],
    execution_constraints: dict[str, str] | None = None,
) -> PolicyDecisionV1:
    gate = policy_gate or _policy_gate_for_decision(decision, candidate.required_policy_gate)
    constraints = dict(execution_constraints or {})
    constraints.setdefault("layer", "9_deferred")
    if decision == "requires_operator_review":
        constraints["requires_operator"] = "true"
    constraints["max_scope"] = allowed_scope
    return PolicyDecisionV1(
        decision_id=stable_policy_decision_id(
            proposal_id=candidate.proposal_id,
            policy_id=policy.policy_id,
        ),
        proposal_id=candidate.proposal_id,
        decision=decision,
        policy_gate=gate,
        autonomy_tier=autonomy_tier,
        risk_score=clamp01(risk_score),
        reversibility_score=clamp01(reversibility_score),
        confidence_score=clamp01(confidence_score),
        allowed_scope=allowed_scope,
        reasons=reasons,
        evidence_refs=sorted(set(evidence_refs)),
        blocked_by=blocked_by,
        execution_constraints=constraints,
    )


def evaluate_proposal_candidate(
    *,
    candidate: ProposalCandidateV1,
    proposal_frame: ProposalFrameV1,
    self_state: SelfStateV1,
    policy: SubstratePolicyV1,
) -> PolicyDecisionV1:
    del self_state  # reserved for future context-aware rules
    rule = kind_rule(policy, candidate.proposal_kind)
    allowed_scope = rule.allowed_scope if rule else "none"
    autonomy_tier = rule.max_autonomy_tier if rule else policy.autonomy.default_tier
    reasons: list[str] = []
    evidence_refs = list(candidate.evidence_refs)
    evidence_refs.extend(
        [
            f"proposal_frame:{proposal_frame.frame_id}",
            f"self_state:{proposal_frame.source_self_state_id}",
        ]
    )

    decision: DecisionLiteral = (
        rule.default_decision if rule else "deferred"  # type: ignore[assignment]
    )

    blocked = hard_block_hits(candidate, policy)
    if blocked:
        return _finish(
            candidate=candidate,
            policy=policy,
            decision="rejected",
            policy_gate="execution_policy",
            autonomy_tier="observe_only",
            allowed_scope="none",
            risk_score=candidate.risk_score,
            reversibility_score=candidate.reversibility_score,
            confidence_score=candidate.confidence_score,
            reasons=["hard_block_execution_intent"],
            evidence_refs=evidence_refs,
            blocked_by=blocked,
            execution_constraints={"layer": "9_deferred", "blocked": "true"},
        )

    if candidate.proposal_kind == "defer":
        decision = "deferred"
        reasons.append("proposal_kind_defer")
    elif candidate.risk_score >= policy.thresholds.reject_above_risk:
        decision = "rejected"
        reasons.append("risk_above_reject_threshold")
    elif candidate.risk_score >= policy.thresholds.defer_above_risk:
        decision = "deferred"
        reasons.append("risk_above_defer_threshold")
    elif candidate.proposal_kind == "prepare_action":
        decision = "requires_operator_review"
        reasons.append("prepare_action_never_auto_execute_v1")
    elif candidate.required_policy_gate == "operator_review":
        decision = "requires_operator_review"
        reasons.append("candidate_requires_operator_review")
    elif candidate.risk_score >= policy.thresholds.require_review_above_risk:
        decision = "requires_operator_review"
        reasons.append("risk_above_review_threshold")
    elif candidate.reversibility_score < policy.thresholds.require_review_below_reversibility:
        decision = "requires_operator_review"
        reasons.append("reversibility_below_threshold")
    elif candidate.confidence_score < policy.thresholds.require_review_below_confidence:
        decision = "requires_operator_review"
        reasons.append("confidence_below_threshold")
    elif candidate.required_policy_gate in ("autonomy_policy", "execution_policy"):
        decision = "requires_operator_review"
        reasons.append("elevated_policy_gate_required")
    elif (
        is_read_only_candidate(candidate, policy)
        and candidate.risk_score <= policy.thresholds.approve_read_only_max_risk
        and candidate.required_policy_gate in ("none", "read_only")
    ):
        decision = "approved_read_only"
        reasons.append("read_only_low_risk")
    elif rule is not None:
        decision = rule.default_decision  # type: ignore[assignment]
        reasons.append("proposal_kind_default")

    if decision == "approved_for_execution" and not policy.autonomy.allow_execution_without_operator:
        decision = "requires_operator_review"
        reasons.append("execution_without_operator_disabled")

    return _finish(
        candidate=candidate,
        policy=policy,
        decision=decision,
        policy_gate=None,
        autonomy_tier=autonomy_tier,
        allowed_scope=allowed_scope,
        risk_score=candidate.risk_score,
        reversibility_score=candidate.reversibility_score,
        confidence_score=candidate.confidence_score,
        reasons=reasons,
        evidence_refs=evidence_refs,
        blocked_by=[],
    )
