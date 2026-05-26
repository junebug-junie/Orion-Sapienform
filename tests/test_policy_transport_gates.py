from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.policy.evaluator import evaluate_proposal_candidate
from orion.policy.policy import load_substrate_policy
from orion.proposals.templates import cast_policy_gate, cast_proposal_kind, cast_proposed_effect, cast_target_kind
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)
POLICY = load_substrate_policy(REPO_ROOT / "config" / "policy" / "substrate_policy.v1.yaml")


def _candidate(*, proposal_id: str, kind: str, execution_note: str = "") -> ProposalCandidateV1:
    return ProposalCandidateV1(
        proposal_id=proposal_id,
        title="t",
        description="d",
        target_id="capability:transport",
        target_kind=cast_target_kind("capability"),
        priority_score=0.5,
        urgency_score=0.3,
        confidence_score=0.8,
        risk_score=0.05,
        reversibility_score=1.0,
        proposal_kind=cast_proposal_kind(kind),
        proposed_effect=cast_proposed_effect("increase_observability"),
        required_policy_gate=cast_policy_gate("read_only"),
        execution_intent={"template": "inspect_transport_status", "note": execution_note},
    )


def _self_state_minimal() -> "SelfStateV1":
    from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

    return SelfStateV1(
        self_state_id="self.state:test",
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:test",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.5,
        overall_confidence=0.7,
        dimensions={
            "coherence": SelfStateDimensionV1(dimension_id="coherence", score=0.8, confidence=0.7)
        },
    )


def _frame() -> ProposalFrameV1:
    state = _self_state_minimal()
    return ProposalFrameV1(
        frame_id="proposal.frame:test",
        generated_at=NOW,
        source_self_state_id=state.self_state_id,
        source_self_state_generated_at=state.generated_at,
        source_attention_frame_id=state.source_attention_frame_id,
        source_field_tick_id=state.source_field_tick_id,
        overall_action_pressure=0.5,
        overall_risk=0.1,
        candidates=[],
    )


def test_read_only_transport_inspect_approved() -> None:
    self_state = _self_state_minimal()
    decision = evaluate_proposal_candidate(
        candidate=_candidate(proposal_id="proposal:inspect_transport_status:s", kind="inspect"),
        proposal_frame=_frame(),
        self_state=self_state,
        policy=POLICY,
    )
    assert decision.decision == "approved_read_only"


def test_restart_bus_rejected() -> None:
    self_state = _self_state_minimal()
    decision = evaluate_proposal_candidate(
        candidate=_candidate(
            proposal_id="proposal:restart_bus:s",
            kind="prepare_action",
            execution_note="restart_bus",
        ),
        proposal_frame=_frame(),
        self_state=self_state,
        policy=POLICY,
    )
    assert decision.decision == "rejected"
