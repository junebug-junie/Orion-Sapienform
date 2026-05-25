from datetime import datetime, timezone
from pathlib import Path

from orion.self_state.policy import load_self_state_policy
from orion.self_state.scoring import (
    agency_readiness_score,
    collect_field_channel_pressures,
    condition_from_intensity,
    coherence_score,
    map_channels_to_dimensions,
    uncertainty_score,
)
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_self_state_policy(REPO / "config" / "self_state" / "self_state_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _field_high_execution() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_scoring",
        node_vectors={"node:athena": {"execution_load": 1.0, "execution_friction": 0.0}},
        capability_vectors={},
    )


def test_high_execution_load_raises_execution_pressure() -> None:
    channels = collect_field_channel_pressures(_field_high_execution())
    dims = map_channels_to_dimensions(channel_pressures=channels, policy=POLICY)
    assert dims.get("execution_pressure", 0.0) > 0.5


def test_high_execution_friction_raises_reliability_pressure() -> None:
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_friction",
        node_vectors={"node:athena": {"execution_friction": 1.0}},
    )
    channels = collect_field_channel_pressures(field)
    dims = map_channels_to_dimensions(channel_pressures=channels, policy=POLICY)
    assert dims.get("reliability_pressure", 0.0) > 0.5


def test_high_failure_pressure_raises_reliability() -> None:
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_fail",
        node_vectors={"node:athena": {"failure_pressure": 1.0}},
    )
    dims = map_channels_to_dimensions(
        channel_pressures=collect_field_channel_pressures(field),
        policy=POLICY,
    )
    assert dims.get("reliability_pressure", 0.0) > 0.5


def test_available_capacity_improves_coherence() -> None:
    low = coherence_score(
        channel_pressures={"cpu_pressure": 0.9},
        policy=POLICY,
    )
    high = coherence_score(
        channel_pressures={"available_capacity": 1.0, "confidence": 1.0},
        policy=POLICY,
    )
    assert high > low


def test_high_salience_low_coherence_raises_uncertainty() -> None:
    u = uncertainty_score(overall_salience=1.0, coherence=0.1)
    assert u > 0.5


def test_agency_readiness_falls_with_reliability_pressure() -> None:
    high_rel = agency_readiness_score(
        coherence=0.9,
        execution_pressure=0.2,
        reliability_pressure=0.9,
        uncertainty=0.1,
        resource_pressure=0.1,
    )
    low_rel = agency_readiness_score(
        coherence=0.9,
        execution_pressure=0.2,
        reliability_pressure=0.1,
        uncertainty=0.1,
        resource_pressure=0.1,
    )
    assert low_rel > high_rel


def test_condition_thresholds() -> None:
    t = POLICY.condition_thresholds
    assert condition_from_intensity(0.10, t) == "quiet"
    assert condition_from_intensity(0.30, t) == "steady"
    assert condition_from_intensity(0.55, t) == "loaded"
    assert condition_from_intensity(0.85, t) == "strained"
    assert condition_from_intensity(0.95, t) == "unstable"
