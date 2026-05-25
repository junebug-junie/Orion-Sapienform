from datetime import datetime, timezone
from pathlib import Path

from orion.policy.builder import build_policy_decision_frame, stable_policy_frame_id
from orion.policy.policy import load_substrate_policy
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_substrate_policy(REPO / "config" / "policy" / "substrate_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _loaded_self_state() -> SelfStateV1:
    def dim(dimension_id: str, score: float) -> SelfStateDimensionV1:
        return SelfStateDimensionV1(
            dimension_id=dimension_id,
            score=score,
            confidence=0.9,
        )

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
        dimensions={
            "execution_pressure": dim("execution_pressure", 1.0),
            "reasoning_pressure": dim("reasoning_pressure", 0.9),
            "resource_pressure": dim("resource_pressure", 1.0),
            "agency_readiness": dim("agency_readiness", 0.6),
            "reliability_pressure": dim("reliability_pressure", 0.0),
        },
        summary_labels=["execution_loaded", "resource_pressurized"],
    )


def _candidate(
    proposal_id: str,
    proposal_kind: str,
    *,
    risk_score: float = 0.05,
    required_policy_gate: str = "read_only",
    proposed_effect: str = "increase_observability",
    execution_intent: dict[str, str] | None = None,
) -> ProposalCandidateV1:
    return ProposalCandidateV1(
        proposal_id=proposal_id,
        proposal_kind=proposal_kind,
        title=proposal_id,
        description="test",
        target_id="capability:orchestration",
        target_kind="capability",
        priority_score=0.5,
        urgency_score=0.4,
        confidence_score=0.9,
        risk_score=risk_score,
        reversibility_score=1.0,
        proposed_effect=proposed_effect,
        required_policy_gate=required_policy_gate,
        execution_intent=execution_intent or {"mode": "descriptive_only"},
    )


def _proposal_frame(candidates: list[ProposalCandidateV1]) -> ProposalFrameV1:
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
        candidates=candidates,
    )


def test_builds_policy_decision_frame() -> None:
    state = _loaded_self_state()
    proposal = _proposal_frame(
        [
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
                "proposal:prepare:state",
                "prepare_action",
                required_policy_gate="operator_review",
                proposed_effect="prepare_for_policy_gate",
                risk_score=0.25,
            ),
        ]
    )
    frame = build_policy_decision_frame(
        proposal_frame=proposal,
        self_state=state,
        policy=POLICY,
        now=NOW,
    )
    assert frame.schema_version == "policy.decision.frame.v1"
    assert frame.source_proposal_frame_id == proposal.frame_id
    assert frame.source_self_state_id == state.self_state_id


def test_partitions_decisions() -> None:
    state = _loaded_self_state()
    proposal = _proposal_frame(
        [
            _candidate("proposal:inspect:state", "inspect"),
            _candidate(
                "proposal:prepare:state",
                "prepare_action",
                required_policy_gate="operator_review",
                proposed_effect="prepare_for_policy_gate",
                risk_score=0.25,
            ),
            _candidate(
                "proposal:blocked:state",
                "inspect",
                execution_intent={"mode": "cortex_exec_direct_call"},
            ),
        ]
    )
    frame = build_policy_decision_frame(
        proposal_frame=proposal,
        self_state=state,
        policy=POLICY,
        now=NOW,
    )
    assert any(d.decision == "approved_read_only" for d in frame.approved_decisions)
    assert len(frame.review_required_decisions) >= 1
    assert len(frame.rejected_decisions) >= 1


def test_operator_review_required() -> None:
    state = _loaded_self_state()
    proposal = _proposal_frame(
        [
            _candidate(
                "proposal:prepare:state",
                "prepare_action",
                required_policy_gate="operator_review",
                proposed_effect="prepare_for_policy_gate",
                risk_score=0.25,
            ),
        ]
    )
    frame = build_policy_decision_frame(
        proposal_frame=proposal,
        self_state=state,
        policy=POLICY,
        now=NOW,
    )
    assert frame.operator_review_required is True


def test_execution_allowed_false_in_v1() -> None:
    state = _loaded_self_state()
    proposal = _proposal_frame([_candidate("proposal:inspect:state", "inspect")])
    frame = build_policy_decision_frame(
        proposal_frame=proposal,
        self_state=state,
        policy=POLICY,
        now=NOW,
    )
    assert frame.execution_allowed is False


def test_stable_frame_id() -> None:
    state = _loaded_self_state()
    proposal = _proposal_frame([_candidate("proposal:inspect:state", "inspect")])
    expected = stable_policy_frame_id(
        proposal_frame_id=proposal.frame_id,
        policy_id=POLICY.policy_id,
    )
    frame = build_policy_decision_frame(
        proposal_frame=proposal,
        self_state=state,
        policy=POLICY,
        now=NOW,
    )
    assert frame.frame_id == expected
