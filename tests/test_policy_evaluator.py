from datetime import datetime, timezone
from pathlib import Path

from orion.policy.evaluator import evaluate_proposal_candidate
from orion.policy.policy import load_substrate_policy
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_substrate_policy(REPO / "config" / "policy" / "substrate_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:test",
        generated_at=NOW,
        source_field_tick_id="tick:test",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame:test",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.9,
    )


def _proposal_frame() -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id="proposal.frame:test:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id="self.state:test",
        source_self_state_generated_at=NOW,
        source_attention_frame_id="frame:test",
        source_field_tick_id="tick:test",
        overall_action_pressure=0.5,
        overall_risk=0.1,
    )


def _candidate(**overrides) -> ProposalCandidateV1:
    base = dict(
        proposal_id="proposal:test:state",
        proposal_kind="inspect",
        title="Inspect",
        description="Desc",
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
    base.update(overrides)
    return ProposalCandidateV1(**base)


def test_read_only_inspect_low_risk_approved() -> None:
    result = evaluate_proposal_candidate(
        candidate=_candidate(),
        proposal_frame=_proposal_frame(),
        self_state=_self_state(),
        policy=POLICY,
    )
    assert result.decision == "approved_read_only"


def test_summarize_low_risk_approved_read_only() -> None:
    result = evaluate_proposal_candidate(
        candidate=_candidate(
            proposal_kind="summarize",
            target_kind="self_state",
            target_id="self:current",
        ),
        proposal_frame=_proposal_frame(),
        self_state=_self_state(),
        policy=POLICY,
    )
    assert result.decision == "approved_read_only"


def test_prepare_action_requires_operator_review() -> None:
    result = evaluate_proposal_candidate(
        candidate=_candidate(
            proposal_kind="prepare_action",
            required_policy_gate="operator_review",
            risk_score=0.25,
            proposed_effect="prepare_for_policy_gate",
        ),
        proposal_frame=_proposal_frame(),
        self_state=_self_state(),
        policy=POLICY,
    )
    assert result.decision == "requires_operator_review"


def test_high_risk_rejected() -> None:
    result = evaluate_proposal_candidate(
        candidate=_candidate(risk_score=0.90),
        proposal_frame=_proposal_frame(),
        self_state=_self_state(),
        policy=POLICY,
    )
    assert result.decision == "rejected"


def test_low_reversibility_requires_review() -> None:
    result = evaluate_proposal_candidate(
        candidate=_candidate(reversibility_score=0.30, risk_score=0.10),
        proposal_frame=_proposal_frame(),
        self_state=_self_state(),
        policy=POLICY,
    )
    assert result.decision == "requires_operator_review"


def test_low_confidence_requires_review() -> None:
    result = evaluate_proposal_candidate(
        candidate=_candidate(confidence_score=0.40, risk_score=0.10),
        proposal_frame=_proposal_frame(),
        self_state=_self_state(),
        policy=POLICY,
    )
    assert result.decision == "requires_operator_review"


def test_hard_blocked_execution_intent_rejected() -> None:
    result = evaluate_proposal_candidate(
        candidate=_candidate(execution_intent={"mode": "cortex_exec_direct_call"}),
        proposal_frame=_proposal_frame(),
        self_state=_self_state(),
        policy=POLICY,
    )
    assert result.decision == "rejected"
    assert "cortex_exec_direct_call" in result.blocked_by


def test_no_approved_for_execution_when_disabled() -> None:
    kinds = (
        "observe",
        "inspect",
        "summarize",
        "stabilize",
        "defer",
        "request_policy_review",
        "prepare_action",
    )
    for kind in kinds:
        result = evaluate_proposal_candidate(
            candidate=_candidate(proposal_kind=kind, risk_score=0.05),
            proposal_frame=_proposal_frame(),
            self_state=_self_state(),
            policy=POLICY,
        )
        assert result.decision != "approved_for_execution"
