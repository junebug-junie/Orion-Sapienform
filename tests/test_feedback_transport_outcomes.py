from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.feedback.builder import build_feedback_frame
from orion.feedback.policy import load_feedback_policy
from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)
FEEDBACK_POLICY = load_feedback_policy(REPO_ROOT / "config" / "feedback" / "feedback_policy.v1.yaml")


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:fb",
        generated_at=NOW,
        source_field_tick_id="t",
        source_field_generated_at=NOW,
        source_attention_frame_id="a",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.5,
        overall_confidence=0.7,
        dimensions={
            "coherence": SelfStateDimensionV1(dimension_id="coherence", score=0.8, confidence=0.7)
        },
    )


def test_dry_run_dispatch_yields_dry_run_only() -> None:
    state = _self_state()
    proposal = ProposalFrameV1(
        frame_id="proposal.frame:1",
        generated_at=NOW,
        source_self_state_id=state.self_state_id,
        source_self_state_generated_at=state.generated_at,
        source_attention_frame_id=state.source_attention_frame_id,
        source_field_tick_id=state.source_field_tick_id,
        overall_action_pressure=0.4,
        overall_risk=0.05,
        candidates=[],
    )
    decision = PolicyDecisionV1(
        decision_id="policy.decision:transport:1",
        proposal_id="proposal:inspect_transport_status:s",
        decision="approved_read_only",
        policy_gate="read_only",
        autonomy_tier="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.8,
        allowed_scope="inspect_only",
    )
    policy_frame = PolicyDecisionFrameV1(
        frame_id="policy.frame:1",
        generated_at=NOW,
        source_proposal_frame_id=proposal.frame_id,
        source_self_state_id=state.self_state_id,
        decisions=[decision],
        approved_decisions=[decision],
        overall_risk=0.05,
    )
    dispatch = ExecutionDispatchFrameV1(
        frame_id="dispatch.frame:1",
        generated_at=NOW,
        source_policy_frame_id=policy_frame.frame_id,
        source_proposal_frame_id=proposal.frame_id,
        source_self_state_id=state.self_state_id,
        dispatch_mode="dry_run",
        dispatch_attempted=False,
        dispatch_count=0,
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state_before=state,
        self_state_after=state,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "dry_run_only"
    assert frame.observations
