from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def test_proposal_candidate_validates() -> None:
    c = ProposalCandidateV1(
        proposal_id="proposal:inspect:state1",
        proposal_kind="inspect",
        title="Inspect",
        description="Desc",
        target_id="capability:orchestration",
        target_kind="capability",
        priority_score=0.5,
        urgency_score=0.4,
        confidence_score=0.8,
        risk_score=0.05,
        reversibility_score=1.0,
        proposed_effect="increase_observability",
        required_policy_gate="read_only",
    )
    assert c.proposal_kind == "inspect"


def test_proposal_frame_validates() -> None:
    frame = ProposalFrameV1(
        frame_id="proposal.frame:state1:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id="self.state:1",
        source_self_state_generated_at=NOW,
        source_attention_frame_id="frame:1",
        source_field_tick_id="tick:1",
        overall_action_pressure=0.4,
        overall_risk=0.1,
    )
    assert frame.schema_version == "proposal.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ProposalCandidateV1(
            proposal_id="p1",
            proposal_kind="observe",
            title="T",
            description="D",
            target_id="t",
            target_kind="system",
            priority_score=0.1,
            urgency_score=0.1,
            confidence_score=0.1,
            risk_score=0.1,
            reversibility_score=1.0,
            proposed_effect="no_effect",
            extra_field=True,
        )


def test_score_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        ProposalCandidateV1(
            proposal_id="p1",
            proposal_kind="observe",
            title="T",
            description="D",
            target_id="t",
            target_kind="system",
            priority_score=1.5,
            urgency_score=0.1,
            confidence_score=0.1,
            risk_score=0.1,
            reversibility_score=1.0,
            proposed_effect="no_effect",
        )


def test_roundtrip_json() -> None:
    frame = ProposalFrameV1(
        frame_id="proposal.frame:state1:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id="self.state:1",
        source_self_state_generated_at=NOW,
        source_attention_frame_id="frame:1",
        source_field_tick_id="tick:1",
        overall_action_pressure=0.4,
        overall_risk=0.1,
        candidates=[
            ProposalCandidateV1(
                proposal_id="proposal:inspect:state1",
                proposal_kind="inspect",
                title="Inspect",
                description="Desc",
                target_id="capability:orchestration",
                target_kind="capability",
                priority_score=0.5,
                urgency_score=0.4,
                confidence_score=0.8,
                risk_score=0.05,
                reversibility_score=1.0,
                proposed_effect="increase_observability",
            )
        ],
    )
    payload = frame.model_dump(mode="json")
    restored = ProposalFrameV1.model_validate(payload)
    assert restored.frame_id == frame.frame_id
    assert len(restored.candidates) == 1
