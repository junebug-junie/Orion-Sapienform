from datetime import datetime, timezone

from orion.feedback.extractors import (
    classify_pressure_deltas,
    extract_field_pressure_snapshot,
    pressure_delta,
)
from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
CHANNELS = ["execution_pressure", "reasoning_pressure", "resource_pressure", "reliability_pressure"]


def _field(node_vectors: dict[str, dict[str, float]]) -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="field.tick:test",
        node_vectors=node_vectors,
    )


def test_extract_snapshot() -> None:
    snap = extract_field_pressure_snapshot(
        _field({"node:test": {"execution_pressure": 1.0, "thermal_pressure": 0.2}}),
        CHANNELS,
    )
    assert snap["execution_pressure"] == 1.0
    assert snap.get("resource_pressure", 0.0) == 0.2
    assert snap.get("reliability_pressure", 0.0) == 0.0


def test_extract_snapshot_none_field() -> None:
    snap = extract_field_pressure_snapshot(None, CHANNELS)
    assert snap == {ch: 0.0 for ch in CHANNELS}


def test_pressure_delta() -> None:
    before = {"execution_pressure": 1.0, "reliability_pressure": 0.2}
    after = {"execution_pressure": 0.5, "reliability_pressure": 0.6}
    delta = pressure_delta(before, after)
    assert abs(delta["execution_pressure"] - (-0.5)) < 1e-6
    assert abs(delta["reliability_pressure"] - 0.4) < 1e-6


def test_classify_positive_negative() -> None:
    delta = {"reliability_pressure": 0.2, "execution_pressure": -0.3, "resource_pressure": 0.1}
    pos, neg = classify_pressure_deltas(
        delta,
        {
            "reliability_pressure": "decrease",
            "execution_pressure": "decrease",
            "resource_pressure": "decrease",
        },
    )
    assert "execution_pressure" in pos[0]
    assert "reliability_pressure" in neg[0]
    assert "resource_pressure" in neg[1]
