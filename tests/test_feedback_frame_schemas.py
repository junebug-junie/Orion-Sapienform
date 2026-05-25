from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.feedback_frame import FeedbackFrameV1, OutcomeObservationV1

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)


def test_outcome_observation_validates() -> None:
    obs = OutcomeObservationV1(
        observation_id="obs:dispatch:d1:dry_run",
        source_kind="dispatch_candidate",
        source_id="dispatch:proposal:inspect:feedback_policy.v1",
        outcome_kind="dry_run",
        score=0.5,
        confidence=0.9,
        observed_at=NOW,
    )
    assert obs.outcome_kind == "dry_run"


def test_feedback_frame_validates() -> None:
    obs = OutcomeObservationV1(
        observation_id="obs:dispatch:d1:dry_run",
        source_kind="dispatch_candidate",
        source_id="dispatch:proposal:inspect:feedback_policy.v1",
        outcome_kind="dry_run",
        score=0.5,
        confidence=0.9,
        observed_at=NOW,
    )
    frame = FeedbackFrameV1(
        frame_id="feedback.frame:execution.dispatch.frame:pf1:feedback_policy.v1",
        generated_at=NOW,
        source_execution_dispatch_frame_id="execution.dispatch.frame:pf1:execution_dispatch_policy.v1",
        source_policy_frame_id="policy.frame:pf1:substrate_policy.v1",
        source_proposal_frame_id="proposal.frame:pf1:proposal_policy.v1",
        source_self_state_id="self.state:pf1",
        outcome_status="dry_run_only",
        outcome_score=0.5,
        confidence_score=0.9,
        observations=[obs],
    )
    assert frame.schema_version == "feedback.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        OutcomeObservationV1(
            observation_id="o1",
            source_kind="dispatch_candidate",
            source_id="s1",
            outcome_kind="dry_run",
            score=0.5,
            confidence=0.9,
            observed_at=NOW,
            extra_field=True,
        )


def test_score_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        FeedbackFrameV1(
            frame_id="f1",
            generated_at=NOW,
            source_execution_dispatch_frame_id="d1",
            outcome_score=1.5,
            confidence_score=0.9,
        )


def test_roundtrip_json() -> None:
    frame = FeedbackFrameV1(
        frame_id="feedback.frame:execution.dispatch.frame:pf1:feedback_policy.v1",
        generated_at=NOW,
        source_execution_dispatch_frame_id="execution.dispatch.frame:pf1:execution_dispatch_policy.v1",
        outcome_status="unknown",
        outcome_score=0.25,
        confidence_score=0.5,
    )
    restored = FeedbackFrameV1.model_validate(frame.model_dump(mode="json"))
    assert restored.frame_id == frame.frame_id
