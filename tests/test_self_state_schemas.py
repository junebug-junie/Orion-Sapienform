from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def test_self_state_dimension_v1_validates() -> None:
    d = SelfStateDimensionV1(
        dimension_id="execution_pressure",
        score=0.8,
        confidence=0.7,
        dominant_evidence=["node:athena.execution_load"],
        reasons=["execution_load elevated"],
    )
    assert d.dimension_id == "execution_pressure"


def test_self_state_v1_roundtrip() -> None:
    state = SelfStateV1(
        self_state_id="self.state:tick_a:frame_a:self_state_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_a",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_a:field_attention_policy.v1",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.6,
    )
    payload = state.model_dump(mode="json")
    restored = SelfStateV1.model_validate(payload)
    assert restored.schema_version == "self.state.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        SelfStateDimensionV1(
            dimension_id="coherence",
            score=0.5,
            confidence=0.5,
            bogus=True,  # type: ignore[call-arg]
        )


def test_scores_reject_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SelfStateDimensionV1(
            dimension_id="coherence",
            score=1.5,
            confidence=0.5,
        )
