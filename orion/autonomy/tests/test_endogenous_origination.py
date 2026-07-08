"""Unit tests for the deterministic endogenous-origination engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.autonomy.endogenous_origination import (
    OriginationConfig,
    OriginationEngine,
)
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


def _make_self_state(
    *,
    trajectory: dict | None = None,
    dwell_ticks: int = 0,
    unresolved: list | None = None,
    agency: float | None = None,
    overall_intensity: float = 0.0,
    dimensions: dict | None = None,
) -> SelfStateV1:
    """Construct a minimal-valid SelfStateV1 with the fields this engine reads."""
    dims: dict = {}
    if dimensions is not None:
        dims = dimensions
    elif agency is not None:
        dims = {
            "agency_readiness": SelfStateDimensionV1(
                dimension_id="agency_readiness", score=agency, confidence=1.0
            )
        }
    return SelfStateV1(
        self_state_id="ss1",
        generated_at=NOW,
        source_field_tick_id="ft1",
        source_field_generated_at=NOW,
        source_attention_frame_id="af1",
        source_attention_generated_at=NOW,
        overall_intensity=overall_intensity,
        overall_confidence=0.7,
        dimensions=dims,
        dimension_trajectory=trajectory or {},
        attention_dwell_ticks=dwell_ticks,
        unresolved_pressures=unresolved or [],
    )


def _fill_ring(engine: OriginationEngine, state: SelfStateV1, n: int) -> None:
    for _ in range(n):
        engine.observe(state)


# 1. High drift + empty exogenous window -> fires an endogenous tension.
def test_fires_on_high_drift_empty_exogenous_window() -> None:
    engine = OriginationEngine()
    # drift=1.0 (traj), agency=0.6 & intensity=0 -> A=0.6, W=0
    # P = 0.4*1 + 0.35*0 + 0.25*0.6 = 0.55 >= threshold; drift dominant -> coherence
    state = _make_self_state(trajectory={"coherence": 1.0}, agency=0.6, overall_intensity=0.0)
    _fill_ring(engine, state, 4)

    ev = engine.maybe_originate(exogenous_tension_count=0, now=NOW)
    assert ev is not None
    assert ev.origin == "endogenous"
    assert ev.drive_impacts == {"coherence": 1.0}
    assert ev.magnitude <= engine.cfg.mag_cap
    assert ev.origination_signal
    assert set(ev.origination_signal) == {"drift", "dwell", "agency", "P"}
    assert ev.kind == "tension.endogenous.v1"
    assert ev.provenance.intake_channel == "substrate.self_state.v1"


# 2. Exogenous input present -> never fires regardless of P.
def test_exogenous_present_suppresses() -> None:
    engine = OriginationEngine()
    # Maximal P snapshot.
    state = _make_self_state(
        trajectory={"a": 1.0}, dwell_ticks=100, unresolved=["p1", "p2", "p3", "p4"], agency=1.0
    )
    _fill_ring(engine, state, 8)

    ev = engine.maybe_originate(exogenous_tension_count=5, now=NOW)
    assert ev is None
    assert engine.last_signal["P"] >= engine.cfg.threshold  # P was high, still suppressed
    assert engine.last_signal["fired"] is False


# 3. Cooldown: fire, immediate re-call -> None, after cooldown+1 -> fires.
def test_cooldown_gate() -> None:
    engine = OriginationEngine()
    state = _make_self_state(trajectory={"a": 1.0}, agency=0.6)
    _fill_ring(engine, state, 4)

    first = engine.maybe_originate(exogenous_tension_count=0, now=NOW)
    assert first is not None

    # Same instant -> cooldown blocks.
    again = engine.maybe_originate(exogenous_tension_count=0, now=NOW)
    assert again is None
    assert engine.last_signal["fired"] is False

    # After cooldown elapses -> fires again.
    later = NOW + timedelta(seconds=engine.cfg.cooldown_sec + 1)
    third = engine.maybe_originate(exogenous_tension_count=0, now=later)
    assert third is not None


# 4. Dominance / override mapping table.
@pytest.mark.parametrize(
    "state_kwargs, expected_drive",
    [
        # drift-dominant -> coherence: D=1, W=0, A=0.6 ; P=0.4+0.15=0.55
        (dict(trajectory={"a": 1.0}, agency=0.6, overall_intensity=0.0), "coherence"),
        # dwell-dominant -> autonomy: D=0.5(drift_w=0.2), W=1(dwell_w=0.35), A=0
        (
            dict(
                trajectory={"a": 0.5},
                dwell_ticks=20,
                unresolved=["p1", "p2", "p3", "p4"],
                overall_intensity=1.0,
            ),
            "autonomy",
        ),
        # agency-dominant -> capability: D=0.6(0.24), W=0.5(0.175), A=1(0.25)
        (dict(trajectory={"a": 0.6}, dwell_ticks=20, agency=1.0, overall_intensity=0.0), "capability"),
        # social_pressure override -> relational
        (
            dict(trajectory={"a": 1.0}, dwell_ticks=100, unresolved=["social_pressure"], agency=1.0),
            "relational",
        ),
        # continuity_pressure override -> continuity
        (
            dict(
                trajectory={"a": 1.0},
                dwell_ticks=100,
                unresolved=["continuity_pressure"],
                agency=1.0,
            ),
            "continuity",
        ),
    ],
)
def test_drive_mapping_table(state_kwargs, expected_drive) -> None:
    engine = OriginationEngine()
    state = _make_self_state(**state_kwargs)
    _fill_ring(engine, state, 4)
    ev = engine.maybe_originate(exogenous_tension_count=0, now=NOW)
    assert ev is not None, f"expected fire for {expected_drive}, signal={engine.last_signal}"
    assert ev.drive_impacts == {expected_drive: 1.0}


# 5. Adversarial P: all sub-signals maxed -> magnitude clamped at mag_cap.
def test_magnitude_capped() -> None:
    engine = OriginationEngine()
    state = _make_self_state(
        trajectory={"a": 1.0}, dwell_ticks=100, unresolved=["p1", "p2", "p3", "p4"], agency=1.0
    )
    _fill_ring(engine, state, 8)
    ev = engine.maybe_originate(exogenous_tension_count=0, now=NOW)
    assert ev is not None
    assert engine.last_signal["P"] == pytest.approx(1.0)
    assert ev.magnitude <= engine.cfg.mag_cap
    assert ev.magnitude == pytest.approx(engine.cfg.mag_cap)


# 6. Cold/empty ring -> None, no raise.
def test_cold_ring_returns_none() -> None:
    engine = OriginationEngine()
    ev = engine.maybe_originate(exogenous_tension_count=0, now=NOW)
    assert ev is None
    assert engine.last_signal["fired"] is False


# 7. Malformed self_state -> observe does not raise; no fire, no raise.
def test_malformed_self_state_never_raises() -> None:
    engine = OriginationEngine()

    # Missing agency dim + empty trajectory (valid but sparse).
    engine.observe(_make_self_state(trajectory={}, dimensions={}))

    # Non-SelfStateV1 garbage objects -> skipped silently.
    class _Garbage:
        dimension_trajectory = "not a dict"
        dimensions = None
        overall_intensity = "nan"
        attention_dwell_ticks = "x"
        unresolved_pressures = 12345

    engine.observe(_Garbage())  # type: ignore[arg-type]
    engine.observe(None)  # type: ignore[arg-type]

    ev = engine.maybe_originate(exogenous_tension_count=0, now=NOW)
    assert ev is None  # sparse/low-P, no fire
    assert isinstance(engine.last_signal, dict)


# 8. last_signal reflects the last computation.
def test_last_signal_shape() -> None:
    engine = OriginationEngine()
    state = _make_self_state(trajectory={"a": 1.0}, agency=0.6)
    _fill_ring(engine, state, 4)
    engine.maybe_originate(exogenous_tension_count=0, now=NOW)
    sig = engine.last_signal
    assert set(sig) == {"drift", "dwell", "agency", "P", "fired"}
    assert sig["fired"] is True
    assert sig["drift"] == pytest.approx(1.0)
    # last_signal is a copy, not a live reference.
    sig["P"] = -999
    assert engine.last_signal["P"] != -999


# Bounded ring: never exceeds cfg.window.
def test_ring_bounded() -> None:
    cfg = OriginationConfig(window=3)
    engine = OriginationEngine(cfg)
    state = _make_self_state(trajectory={"a": 0.5})
    _fill_ring(engine, state, 50)
    assert len(engine._ring) == 3
