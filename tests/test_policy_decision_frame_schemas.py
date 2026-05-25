from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def test_policy_decision_validates() -> None:
    d = PolicyDecisionV1(
        decision_id="policy.decision:proposal:inspect:state:substrate_policy.v1",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
        allowed_scope="inspect_only",
        reasons=["read_only_low_risk"],
    )
    assert d.decision == "approved_read_only"


def test_policy_decision_frame_validates() -> None:
    decision = PolicyDecisionV1(
        decision_id="policy.decision:proposal:inspect:state:substrate_policy.v1",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
    )
    frame = PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:state:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id="proposal.frame:state:proposal_policy.v1",
        source_self_state_id="self.state:state",
        decisions=[decision],
        approved_decisions=[decision],
        overall_risk=0.05,
    )
    assert frame.schema_version == "policy.decision.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        PolicyDecisionV1(
            decision_id="d1",
            proposal_id="p1",
            decision="rejected",
            policy_gate="none",
            risk_score=0.1,
            reversibility_score=1.0,
            confidence_score=0.9,
            extra_field=True,
        )


def test_score_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        PolicyDecisionV1(
            decision_id="d1",
            proposal_id="p1",
            decision="rejected",
            policy_gate="none",
            risk_score=1.5,
            reversibility_score=1.0,
            confidence_score=0.9,
        )


def test_roundtrip_json() -> None:
    decision = PolicyDecisionV1(
        decision_id="policy.decision:p1:substrate_policy.v1",
        proposal_id="p1",
        decision="requires_operator_review",
        policy_gate="operator_review",
        risk_score=0.3,
        reversibility_score=0.4,
        confidence_score=0.6,
        reasons=["low_reversibility"],
    )
    frame = PolicyDecisionFrameV1(
        frame_id="policy.frame:pf1:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id="pf1",
        source_self_state_id="ss1",
        decisions=[decision],
        review_required_decisions=[decision],
        overall_risk=0.3,
        operator_review_required=True,
    )
    payload = frame.model_dump(mode="json")
    restored = PolicyDecisionFrameV1.model_validate(payload)
    assert restored.frame_id == frame.frame_id
