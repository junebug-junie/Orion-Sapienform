"""O3: predictive drive's first primary tension source, grounded on
self_state.overall_surprise (orion/autonomy/drives_and_autonomy_retrospective.md §5a).

Covers extract_tensions_from_self_state's new tension.prediction_surprise.v1
block: fires above the 0.30 absolute threshold, scales by trajectory
multiplier, clamps at 1.0, and stays silent below threshold / at default 0.0.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from orion.spark.concept_induction.tensions import extract_tensions_from_self_state

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


def _envelope() -> BaseEnvelope:
    return BaseEnvelope(
        id=uuid4(), kind="substrate.self_state.v1", correlation_id=uuid4(),
        created_at=NOW, source=ServiceRef(name="orion-substrate-runtime", version="0.1.0", node="athena"),
        payload={},
    )


def _self_state(*, overall_surprise: float | None = None, trajectory_condition: str | None = None) -> SelfStateV1:
    # Neutral dimension values so the other blocks in
    # extract_tensions_from_self_state don't fire and confound assertions
    # about the new prediction_surprise block specifically.
    dims = {
        "coherence": SelfStateDimensionV1(dimension_id="coherence", score=0.5, confidence=1.0),
        "agency_readiness": SelfStateDimensionV1(dimension_id="agency_readiness", score=0.9, confidence=1.0),
        "social_pressure": SelfStateDimensionV1(dimension_id="social_pressure", score=0.1, confidence=1.0),
        "uncertainty": SelfStateDimensionV1(dimension_id="uncertainty", score=0.2, confidence=1.0),
        "resource_pressure": SelfStateDimensionV1(dimension_id="resource_pressure", score=0.2, confidence=1.0),
        "execution_pressure": SelfStateDimensionV1(dimension_id="execution_pressure", score=0.2, confidence=1.0),
    }
    kwargs = dict(
        self_state_id=str(uuid4()), generated_at=NOW,
        source_field_tick_id="ft", source_field_generated_at=NOW,
        source_attention_frame_id="af", source_attention_generated_at=NOW,
        overall_intensity=0.3, overall_confidence=0.7, dimensions=dims,
    )
    if trajectory_condition is not None:
        kwargs["trajectory_condition"] = trajectory_condition
    if overall_surprise is not None:
        kwargs["overall_surprise"] = overall_surprise
    return SelfStateV1(**kwargs)


def _prediction_surprise_events(
    self_state: SelfStateV1, *, previous_self_state: SelfStateV1 | None = None
) -> list:
    out = extract_tensions_from_self_state(
        envelope=_envelope(),
        intake_channel="orion:substrate:self_state",
        self_state=self_state,
        previous_self_state=previous_self_state,
    )
    return [e for e in out if e.kind == "tension.prediction_surprise.v1"]


def test_surprise_above_threshold_fires_with_expected_magnitude() -> None:
    ss = _self_state(overall_surprise=0.5)
    events = _prediction_surprise_events(ss)
    assert len(events) == 1
    ev = events[0]
    assert ev.drive_impacts == {"predictive": 1.0}
    assert 0.0 < ev.magnitude <= 1.0
    # trajectory_condition defaults to "unknown" -> traj_mul == 1.0
    assert ev.magnitude == 0.5


def test_surprise_below_threshold_does_not_fire() -> None:
    ss = _self_state(overall_surprise=0.10)
    events = _prediction_surprise_events(ss)
    assert events == []


def test_surprise_above_threshold_with_degrading_trajectory_scales_magnitude() -> None:
    ss = _self_state(overall_surprise=0.5, trajectory_condition="degrading")
    events = _prediction_surprise_events(ss)
    assert len(events) == 1
    assert events[0].magnitude == 0.625


def test_surprise_at_max_clamps_at_one() -> None:
    ss = _self_state(overall_surprise=1.0)
    events = _prediction_surprise_events(ss)
    assert len(events) == 1
    assert events[0].magnitude == 1.0


def test_surprise_default_zero_does_not_fire() -> None:
    ss = _self_state()  # overall_surprise omitted -> defaults to 0.0
    events = _prediction_surprise_events(ss)
    assert events == []


def test_sustained_elevated_surprise_does_not_refire_every_tick() -> None:
    """Regression: a prior version fired on the absolute level alone, every
    tick, with no gating -- reproducing the leaky integrator's saturation
    pattern (pressure pins near 1.0 within ~5 ticks at bus-tick cadence,
    since decay is negligible against tau=1800s). Once a previous_self_state
    exists, firing must be delta-gated like every sibling block, so a
    sustained-but-unchanging elevated surprise level does not re-fire."""
    prev = _self_state(overall_surprise=0.5)
    now = _self_state(overall_surprise=0.5)  # unchanged tick-to-tick
    events = _prediction_surprise_events(now, previous_self_state=prev)
    assert events == []


def test_surprise_rising_with_previous_state_fires_on_delta_not_absolute() -> None:
    prev = _self_state(overall_surprise=0.10)
    now = _self_state(overall_surprise=0.40)  # delta 0.30 > 0.05 threshold
    events = _prediction_surprise_events(now, previous_self_state=prev)
    assert len(events) == 1
    assert abs(events[0].magnitude - 0.30) < 1e-9
    assert events[0].drive_impacts == {"predictive": 1.0}


def test_surprise_small_delta_below_gate_does_not_fire() -> None:
    prev = _self_state(overall_surprise=0.50)
    now = _self_state(overall_surprise=0.53)  # delta 0.03 < 0.05 threshold
    events = _prediction_surprise_events(now, previous_self_state=prev)
    assert events == []
