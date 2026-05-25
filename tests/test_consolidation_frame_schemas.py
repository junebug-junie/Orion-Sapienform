from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.consolidation_frame import ConsolidationFrameV1, MotifObservationV1

NOW = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)


def test_motif_observation_validates() -> None:
    motif = MotifObservationV1(
        motif_id="motif:loaded_but_reliable:consolidation_policy.v1",
        motif_kind="self_state_pattern",
        label="loaded_but_reliable",
        recurrence_count=3,
        support_score=0.6,
        confidence_score=0.7,
        evidence_frame_ids=["self.state:s1"],
    )
    assert motif.motif_kind == "self_state_pattern"


def test_consolidation_frame_validates() -> None:
    frame = ConsolidationFrameV1(
        frame_id="consolidation.frame:2026-05-25T14:30:00+00:00:2026-05-25T15:30:00+00:00:consolidation_policy.v1",
        generated_at=NOW,
        window_start=datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc),
        window_end=NOW,
        motif_observations=[
            MotifObservationV1(
                motif_id="motif:loaded_but_reliable:consolidation_policy.v1",
                motif_kind="self_state_pattern",
                label="loaded_but_reliable",
                recurrence_count=3,
                support_score=0.6,
                confidence_score=0.7,
            )
        ],
    )
    assert frame.schema_version == "consolidation.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        MotifObservationV1(
            motif_id="m1",
            motif_kind="self_state_pattern",
            label="x",
            recurrence_count=1,
            support_score=0.5,
            confidence_score=0.5,
            extra_field=True,
        )


def test_recurrence_count_min_one() -> None:
    with pytest.raises(ValidationError):
        MotifObservationV1(
            motif_id="m1",
            motif_kind="self_state_pattern",
            label="x",
            recurrence_count=0,
            support_score=0.5,
            confidence_score=0.5,
        )
