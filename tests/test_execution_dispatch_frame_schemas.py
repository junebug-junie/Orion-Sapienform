from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.execution_dispatch_frame import (
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def test_execution_dispatch_candidate_validates() -> None:
    c = ExecutionDispatchCandidateV1(
        dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
        source_decision_id="policy.decision:proposal:inspect:substrate_policy.v1",
        source_proposal_id="proposal:inspect:state",
        dispatch_status="dry_run",
        dispatch_mode="dry_run",
        dispatch_kind="inspect",
        target_id="capability:orchestration",
        target_kind="capability",
        cortex_verb="substrate.inspect",
        cortex_mode="brain",
        risk_score=0.05,
        confidence_score=0.9,
    )
    assert c.dispatch_status == "dry_run"


def test_execution_dispatch_frame_validates() -> None:
    candidate = ExecutionDispatchCandidateV1(
        dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
        source_decision_id="policy.decision:proposal:inspect:substrate_policy.v1",
        source_proposal_id="proposal:inspect:state",
        dispatch_status="dry_run",
        dispatch_mode="dry_run",
        dispatch_kind="inspect",
        target_id="capability:orchestration",
        target_kind="capability",
        risk_score=0.05,
        confidence_score=0.9,
    )
    frame = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:policy.frame:pf1:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:pf1:substrate_policy.v1",
        source_proposal_frame_id="proposal.frame:pf1:proposal_policy.v1",
        source_self_state_id="self.state:pf1",
        candidates=[candidate],
        dispatch_mode="dry_run",
        dispatch_attempted=False,
        dispatch_count=0,
        blocked_count=0,
    )
    assert frame.schema_version == "execution.dispatch.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExecutionDispatchCandidateV1(
            dispatch_id="d1",
            source_decision_id="pd1",
            source_proposal_id="p1",
            dispatch_status="dry_run",
            dispatch_mode="dry_run",
            dispatch_kind="inspect",
            target_id="t1",
            target_kind="capability",
            risk_score=0.1,
            confidence_score=0.9,
            extra_field=True,
        )


def test_score_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionDispatchCandidateV1(
            dispatch_id="d1",
            source_decision_id="pd1",
            source_proposal_id="p1",
            dispatch_status="dry_run",
            dispatch_mode="dry_run",
            dispatch_kind="inspect",
            target_id="t1",
            target_kind="capability",
            risk_score=1.5,
            confidence_score=0.9,
        )


def test_roundtrip_json() -> None:
    frame = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:policy.frame:pf1:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:pf1:substrate_policy.v1",
        source_proposal_frame_id="proposal.frame:pf1:proposal_policy.v1",
        source_self_state_id="self.state:pf1",
        dispatch_mode="dry_run",
    )
    restored = ExecutionDispatchFrameV1.model_validate(frame.model_dump(mode="json"))
    assert restored.frame_id == frame.frame_id
