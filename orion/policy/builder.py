from __future__ import annotations

import logging
from datetime import datetime, timezone

from orion.policy.evaluator import evaluate_proposal_candidate
from orion.policy.policy import SubstratePolicyV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalFrameV1

logger = logging.getLogger(__name__)


def stable_policy_frame_id(*, proposal_frame_id: str, policy_id: str) -> str:
    return f"policy.frame:{proposal_frame_id}:{policy_id}"


def build_policy_decision_frame(
    *,
    proposal_frame: ProposalFrameV1,
    policy: SubstratePolicyV1,
    now: datetime | None = None,
) -> PolicyDecisionFrameV1:
    generated_at = now or datetime.now(timezone.utc)
    # 2026-08-12: per-candidate fault isolation, added after a real incident
    # class was found in review. This was a bare comprehension, so ONE
    # candidate that failed to evaluate raised out of the whole frame. That
    # is not a dropped frame -- it is a permanent stall: the runtime's
    # load_next_proposal_without_policy_frame() selects the oldest proposal
    # frame with no decision frame, so a frame that can never produce one
    # stays the FIFO head forever and every later frame queues behind it.
    # L8 -> L9 -> L10 -> L11 all go dark, with a repeating
    # `policy_runtime_tick_failed` log line as the only symptom.
    #
    # The concrete trigger was `maintenance_bounded` missing from
    # PolicyDecisionV1's Literals (see that file's own note), which is fixed.
    # This exists so the NEXT kind-rule/schema mismatch degrades to one
    # unevaluable candidate instead of stopping cognition -- the deterministic
    # gate, where the Literal fix is only the symptom fix.
    decisions: list[PolicyDecisionV1] = []
    unevaluable: list[str] = []
    for candidate in proposal_frame.candidates:
        try:
            decisions.append(
                evaluate_proposal_candidate(
                    candidate=candidate,
                    proposal_frame=proposal_frame,
                    policy=policy,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately total; see above
            logger.exception(
                "policy_candidate_unevaluable proposal_id=%s kind=%s",
                candidate.proposal_id,
                candidate.proposal_kind,
            )
            # Surfaced on the frame, never silently swallowed: a candidate
            # that cannot be evaluated is a real finding about the policy
            # config, and it must be visible without reading logs.
            unevaluable.append(
                f"candidate_unevaluable:{candidate.proposal_id}:"
                f"{candidate.proposal_kind}:{type(exc).__name__}"
            )
    approved_for_execution = [d for d in decisions if d.decision == "approved_for_execution"]
    approved_read_only = [d for d in decisions if d.decision == "approved_read_only"]
    approved_maintenance = [d for d in decisions if d.decision == "approved_maintenance"]
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
        source_attention_frame_id=proposal_frame.source_attention_frame_id,
        source_field_tick_id=proposal_frame.source_field_tick_id,
        policy_id=policy.policy_id,
        decisions=decisions,
        approved_decisions=approved_for_execution + approved_read_only + approved_maintenance,
        review_required_decisions=review,
        deferred_decisions=deferred,
        rejected_decisions=rejected,
        overall_risk=overall_risk,
        operator_review_required=any(d.decision == "requires_operator_review" for d in decisions),
        execution_allowed=any(d.decision == "approved_for_execution" for d in decisions),
        warnings=list(proposal_frame.warnings) + unevaluable,
    )
