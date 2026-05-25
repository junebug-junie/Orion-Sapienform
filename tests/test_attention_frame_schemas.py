from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1


def test_field_attention_target_v1_validates() -> None:
    t = FieldAttentionTargetV1(
        target_id="node:athena",
        target_kind="node",
        salience_score=0.8,
        pressure_score=0.9,
        novelty_score=0.0,
        urgency_score=0.5,
        confidence_score=0.2,
        dominant_channels={"execution_load": 0.7},
        reasons=["node execution_load is elevated"],
        evidence_refs=["field:tick_abc"],
        suggested_observation_mode="inspect",
    )
    assert t.target_kind == "node"


def test_field_attention_frame_v1_roundtrip() -> None:
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    frame = FieldAttentionFrameV1(
        frame_id="attention.frame:tick_abc:field_attention_policy.v1",
        generated_at=now,
        source_field_tick_id="tick_abc",
        source_field_generated_at=now,
        overall_salience=0.5,
        dominant_targets=[],
    )
    payload = frame.model_dump(mode="json")
    restored = FieldAttentionFrameV1.model_validate(payload)
    assert restored.schema_version == "field.attention.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        FieldAttentionTargetV1(
            target_id="node:athena",
            target_kind="node",
            salience_score=0.5,
            pressure_score=0.5,
            novelty_score=0.0,
            urgency_score=0.0,
            confidence_score=0.0,
            bogus=True,  # type: ignore[call-arg]
        )


def test_scores_reject_out_of_range() -> None:
    with pytest.raises(ValidationError):
        FieldAttentionTargetV1(
            target_id="node:athena",
            target_kind="node",
            salience_score=1.5,
            pressure_score=0.5,
            novelty_score=0.0,
            urgency_score=0.0,
            confidence_score=0.0,
        )
