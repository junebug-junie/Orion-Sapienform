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


def _frame() -> ProposalFrameV1:
    return ProposalFrameV1(
        frame_id="proposal.frame:test",
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:test",
        overall_action_pressure=0.5,
        overall_risk=0.1,
        candidates=[],
    )


def test_read_only_transport_inspect_approved() -> None:
    decision = evaluate_proposal_candidate(
        candidate=_candidate(proposal_id="proposal:inspect_transport_status:s", kind="inspect"),
        proposal_frame=_frame(),
        policy=POLICY,
    )
    assert decision.decision == "approved_read_only"


def test_restart_bus_rejected() -> None:
    decision = evaluate_proposal_candidate(
        candidate=_candidate(
            proposal_id="proposal:restart_bus:s",
            kind="prepare_action",
            execution_note="restart_bus",
        ),
        proposal_frame=_frame(),
        policy=POLICY,
    )
    assert decision.decision == "rejected"
