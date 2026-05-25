from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.consolidation_frame import ConsolidationFrameV1, ExpectationV1, MotifObservationV1, SparseTensorSliceV1
from orion.schemas.registry import resolve

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


def test_expectation_v1_validates() -> None:
    expectation = ExpectationV1(
        expectation_id="expectation:motif:loaded_but_reliable:consolidation_policy.v1:reliability_clear",
        trigger_motif_id="motif:loaded_but_reliable:consolidation_policy.v1",
        expected_outcome_kind="reliability_clear",
        confidence_score=0.7,
        support_count=3,
        evidence_refs=["self.state:s1"],
        reasons=["derived_from_motif:loaded_but_reliable"],
    )
    assert expectation.expected_outcome_kind == "reliability_clear"


def test_consolidation_frame_includes_expectations() -> None:
    frame = ConsolidationFrameV1(
        frame_id="consolidation.frame:test",
        generated_at=NOW,
        window_start=datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc),
        window_end=NOW,
        expectations=[
            ExpectationV1(
                expectation_id="expectation:test",
                trigger_motif_id="motif:test",
                expected_outcome_kind="unknown",
                confidence_score=0.5,
                support_count=1,
            )
        ],
    )
    assert frame.expectations[0].expectation_id == "expectation:test"


def test_expectation_v1_registered() -> None:
    assert resolve("ExpectationV1") is ExpectationV1


def test_sparse_tensor_slice_v1_validates() -> None:
    tensor = SparseTensorSliceV1(
        tensor_id="tensor:field_attention_self:2026-05-25T14:30:00+00:00:2026-05-25T15:30:00+00:00",
        tensor_kind="field_attention_self",
        axes=["time_bucket", "self_condition", "attention_target", "dimension"],
        coordinates=[
            {
                "time_bucket": "2026-05-25T15:00:00+00:00",
                "self_condition": "loaded",
                "attention_target": "node:athena",
                "dimension": "execution_pressure",
            }
        ],
        values=[0.8],
        evidence_refs=["self.state:s1"],
    )
    assert tensor.tensor_kind == "field_attention_self"
    assert len(tensor.values) == len(tensor.coordinates)


def test_sparse_tensor_slice_v1_registered() -> None:
    assert resolve("SparseTensorSliceV1") is SparseTensorSliceV1
