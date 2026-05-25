from datetime import datetime, timezone

from orion.feedback.extractors import (
    classify_pressure_deltas,
    extract_self_state_pressure_snapshot,
    pressure_delta,
)
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
CHANNELS = ["execution_pressure", "agency_readiness", "uncertainty", "coherence"]


def _state(scores: dict[str, float]) -> SelfStateV1:
    dims = {
        k: SelfStateDimensionV1(dimension_id=k, score=v, confidence=0.9)
        for k, v in scores.items()
    }
    return SelfStateV1(
        self_state_id="self.state:test",
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="att",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.9,
        dimensions=dims,
    )


def test_extract_snapshot() -> None:
    snap = extract_self_state_pressure_snapshot(
        _state({"execution_pressure": 1.0, "agency_readiness": 0.2}),
        CHANNELS,
    )
    assert snap["execution_pressure"] == 1.0
    assert snap.get("coherence", 0.0) == 0.0


def test_pressure_delta() -> None:
    before = {"execution_pressure": 1.0, "agency_readiness": 0.2}
    after = {"execution_pressure": 0.5, "agency_readiness": 0.6}
    delta = pressure_delta(before, after)
    assert abs(delta["execution_pressure"] - (-0.5)) < 1e-6
    assert abs(delta["agency_readiness"] - 0.4) < 1e-6


def test_classify_positive_negative() -> None:
    delta = {"agency_readiness": 0.2, "execution_pressure": -0.3, "uncertainty": 0.1}
    pos, neg = classify_pressure_deltas(
        delta,
        {
            "agency_readiness": "increase",
            "execution_pressure": "decrease",
            "uncertainty": "decrease",
        },
    )
    assert "agency_readiness" in pos[0]
    assert "execution_pressure" in pos[1]
    assert "uncertainty" in neg[0]
