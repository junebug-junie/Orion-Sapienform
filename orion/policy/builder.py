from __future__ import annotations

from datetime import datetime, timezone

from orion.policy.evaluator import evaluate_proposal_candidate
from orion.policy.policy import SubstratePolicyV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.schemas.self_state import SelfStateV1


def stable_policy_frame_id(*, proposal_frame_id: str, policy_id: str) -> str:
    return f"policy.frame:{proposal_frame_id}:{policy_id}"


def build_policy_decision_frame(
    *,
    proposal_frame: ProposalFrameV1,
    self_state: SelfStateV1,
    policy: SubstratePolicyV1,
    now: datetime | None = None,
) -> PolicyDecisionFrameV1:
    generated_at = now or datetime.now(timezone.utc)
    decisions = [
        evaluate_proposal_candidate(
            candidate=candidate,
            proposal_frame=proposal_frame,
            self_state=self_state,
            policy=policy,
        )
        for candidate in proposal_frame.candidates
    ]
    approved_for_execution = [d for d in decisions if d.decision == "approved_for_execution"]
    approved_read_only = [d for d in decisions if d.decision == "approved_read_only"]
    review = [d for d in decisions if d.decision == "requires_operator_review"]
    deferred = [d for d in decisions if d.decision == "deferred"]
    rejected = [d for d in decisions if d.decision == "rejected"]
    overall_risk = max((d.risk_score for d in decisions), default=0.0)
    return PolicyDecisionFrameV1(
        frame_id=stable_policy_frame_id(
            proposal_frame_id=proposal_frame.frame_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_proposal_frame_id=proposal_frame.frame_id,
        source_self_state_id=proposal_frame.source_self_state_id,
        source_attention_frame_id=proposal_frame.source_attention_frame_id,
        source_field_tick_id=proposal_frame.source_field_tick_id,
        policy_id=policy.policy_id,
        decisions=decisions,
        approved_decisions=approved_for_execution + approved_read_only,
        review_required_decisions=review,
        deferred_decisions=deferred,
        rejected_decisions=rejected,
        overall_risk=overall_risk,
        operator_review_required=any(d.decision == "requires_operator_review" for d in decisions),
        execution_allowed=any(d.decision == "approved_for_execution" for d in decisions),
        warnings=list(proposal_frame.warnings),
    )
