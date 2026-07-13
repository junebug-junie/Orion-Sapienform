from datetime import datetime, timezone
from pathlib import Path

from orion.execution_dispatch.builder import (
    build_execution_dispatch_frame,
    stable_execution_dispatch_frame_id,
)
from orion.execution_dispatch.policy import load_execution_dispatch_policy
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_execution_dispatch_policy(
    REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
)
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _loaded_self_state() -> SelfStateV1:
    def dim(dimension_id: str, score: float) -> SelfStateDimensionV1:
        return SelfStateDimensionV1(dimension_id=dimension_id, score=score, confidence=0.9)

    return SelfStateV1(
        self_state_id="self.state:tick_live:frame_live:self_state_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_live",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_live:field_attention_policy.v1",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.655,
        overall_confidence=0.9,
        dimensions={"execution_pressure": dim("execution_pressure", 1.0)},
        summary_labels=["execution_loaded"],
    )


def _candidate(proposal_id: str, proposal_kind: str, **kwargs) -> ProposalCandidateV1:
    base = dict(
        proposal_id=proposal_id,
        proposal_kind=proposal_kind,
        title=proposal_id,
        description="test",
        target_id="capability:orchestration",
        target_kind="capability",
        priority_score=0.5,
        urgency_score=0.4,
        confidence_score=0.9,
        risk_score=0.05,
        reversibility_score=1.0,
        proposed_effect="increase_observability",
        required_policy_gate="read_only",
        execution_intent={"mode": "descriptive_only"},
    )
    base.update(kwargs)
    return ProposalCandidateV1(**base)


def _proposal_frame() -> ProposalFrameV1:
    state = _loaded_self_state()
    return ProposalFrameV1(
        frame_id="proposal.frame:test:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id=state.self_state_id,
        source_self_state_generated_at=state.generated_at,
        source_attention_frame_id=state.source_attention_frame_id,
        source_field_tick_id=state.source_field_tick_id,
        overall_action_pressure=0.6,
        overall_risk=0.3,
        candidates=[
            _candidate("proposal:inspect:state", "inspect"),
            _candidate("proposal:summarize:state", "summarize"),
            _candidate(
                "proposal:review:state",
                "request_policy_review",
                required_policy_gate="operator_review",
                proposed_effect="prepare_for_policy_gate",
                risk_score=0.25,
            ),
            _candidate(
                "proposal:blocked:state",
                "prepare_action",
                required_policy_gate="operator_review",
                proposed_effect="prepare_for_policy_gate",
                risk_score=0.25,
            ),
        ],
    )


def _policy_frame(proposal: ProposalFrameV1) -> PolicyDecisionFrameV1:
    def decision(proposal_id: str, decision_value: str) -> PolicyDecisionV1:
        return PolicyDecisionV1(
            decision_id=f"policy.decision:{proposal_id}:substrate_policy.v1",
            proposal_id=proposal_id,
            decision=decision_value,
            policy_gate="read_only" if decision_value == "approved_read_only" else "operator_review",
            risk_score=0.05 if decision_value == "approved_read_only" else 0.25,
            reversibility_score=1.0,
            confidence_score=0.9,
            allowed_scope="inspect_only" if decision_value == "approved_read_only" else "operator_review_required",
        )

    decisions = [
        decision("proposal:inspect:state", "approved_read_only"),
        decision("proposal:summarize:state", "approved_read_only"),
        decision("proposal:review:state", "requires_operator_review"),
        decision("proposal:blocked:state", "rejected"),
    ]
    approved = [d for d in decisions if d.decision == "approved_read_only"]
    review = [d for d in decisions if d.decision == "requires_operator_review"]
    rejected = [d for d in decisions if d.decision == "rejected"]
    return PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:test:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id=proposal.frame_id,
        source_self_state_id=proposal.source_self_state_id,
        decisions=decisions,
        approved_decisions=approved,
        review_required_decisions=review,
        rejected_decisions=rejected,
        overall_risk=0.3,
        operator_review_required=True,
        execution_allowed=False,
    )


def test_builds_execution_dispatch_frame() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    state = _loaded_self_state()
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=state,
        policy=POLICY,
        now=NOW,
    )
    assert frame.schema_version == "execution.dispatch.frame.v1"
    assert frame.source_policy_frame_id == policy_frame.frame_id
    assert frame.source_proposal_frame_id == proposal.frame_id
    assert frame.source_self_state_id == state.self_state_id


def test_approved_read_only_become_dry_run_candidates() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    kinds = {c.dispatch_kind for c in frame.candidates}
    assert "inspect" in kinds
    assert "summarize" in kinds
    assert all(c.dispatch_status == "dry_run" for c in frame.candidates)
    assert all(c.dispatch_mode == "dry_run" for c in frame.candidates)


def test_review_and_rejected_blocked() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    blocked_ids = {c.source_proposal_id for c in frame.blocked_candidates}
    assert "proposal:review:state" in blocked_ids
    assert "proposal:blocked:state" in blocked_ids
    assert frame.blocked_count >= 2


def test_default_dispatch_mode_and_no_attempt() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    assert frame.dispatch_mode == "dry_run"
    assert frame.dispatch_attempted is False
    assert frame.dispatch_count == 0


def test_no_mutating_scope_in_envelopes() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    for c in frame.candidates:
        constraints = c.request_envelope.get("constraints", {})
        assert constraints.get("read_only") is True


def test_dispatch_read_only_produces_prepared_for_dispatch_not_dispatched() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    policy = POLICY.model_copy(
        update={"mode": POLICY.mode.model_copy(update={"allow_dispatch_read_only": True})}
    )
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=policy,
        now=NOW,
        override_dispatch_mode="dispatch_read_only",
    )

    approved_ids = {"proposal:inspect:state", "proposal:summarize:state"}
    approved_candidates = [c for c in frame.candidates if c.source_proposal_id in approved_ids]
    assert approved_candidates, "expected approved read-only candidates to be present"
    assert all(c.dispatch_status == "prepared_for_dispatch" for c in approved_candidates)

    # Never routed into dispatched_candidates: nothing was actually sent.
    dispatched_ids = {c.source_proposal_id for c in frame.dispatched_candidates}
    assert not (approved_ids & dispatched_ids)
    assert frame.dispatch_count == 0

    all_candidates = frame.candidates + frame.blocked_candidates + frame.dispatched_candidates
    assert all(c.dispatch_status != "dispatched" for c in all_candidates)


def test_stable_frame_id() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    expected = stable_execution_dispatch_frame_id(
        policy_frame_id=policy_frame.frame_id,
        policy_id=POLICY.policy_id,
    )
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    assert frame.frame_id == expected
