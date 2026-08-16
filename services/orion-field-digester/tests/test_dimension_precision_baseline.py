from __future__ import annotations

from datetime import datetime, timezone

from orion.bus.ewma import compute_ewma_update
from orion.field.pressure import field_pressures
from orion.proposals.scoring import DIMENSION_PRECISION_EWMA_ALPHA, DIMENSION_PRECISION_MIN_VARIANCE
from orion.schemas.field_state import FieldStateV1

from app.digestion.precision import update_dimension_precision_baseline

BASE = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def _empty_state(tick_id: str) -> FieldStateV1:
    return FieldStateV1(generated_at=BASE, tick_id=tick_id)


def test_first_tick_seeds_baseline_but_no_zscore_yet() -> None:
    """Regression guard for the same class of bug as recent_perturbations'
    baseline fix: the true "no signal yet" state is an ABSENT zscore key, not
    a fabricated 0.0 -- compute_ewma_update's own documented first-
    observation behavior (orion.bus.ewma.EwmaUpdate.zscore is None when
    prev_count == 0)."""
    state = _empty_state("tick_0")
    state.node_vectors["node:athena"] = {"execution_pressure": 0.3}
    update_dimension_precision_baseline(state)
    assert state.dimension_precision_ewma_n["execution_pressure"] == 1
    assert state.dimension_precision_ewma["execution_pressure"] == 0.3
    assert "execution_pressure" not in state.dimension_precision_zscore


def test_absent_dimension_this_tick_leaves_existing_baseline_untouched() -> None:
    """A dimension with no real channel mapped to it this tick (see
    orion/field/pressure.py::map_channels_to_dimensions) must NOT have its
    baseline updated -- absorbing a fabricated 0.0 would corrupt the real
    trajectory the baseline is tracking."""
    state = _empty_state("tick_0")
    state.node_vectors["node:athena"] = {"execution_pressure": 0.3}
    update_dimension_precision_baseline(state)
    assert state.dimension_precision_ewma_n["execution_pressure"] == 1

    # Tick 2: the channel that maps to execution_pressure is gone this tick.
    state.tick_id = "tick_1"
    state.node_vectors["node:athena"] = {}
    update_dimension_precision_baseline(state)
    assert state.dimension_precision_ewma_n["execution_pressure"] == 1
    assert state.dimension_precision_ewma["execution_pressure"] == 0.3


def test_second_tick_produces_a_real_zscore_matching_compute_ewma_update() -> None:
    """Reuses orion.bus.ewma.compute_ewma_update directly -- this test
    cross-checks the producer's output against an independent call to that
    same function, confirming no reimplementation drift."""
    state = _empty_state("tick_0")
    state.node_vectors["node:athena"] = {"execution_pressure": 0.3}
    update_dimension_precision_baseline(state)

    state.tick_id = "tick_1"
    state.node_vectors["node:athena"] = {"execution_pressure": 0.5}
    update_dimension_precision_baseline(state)

    expected = compute_ewma_update(
        prev_ewma=0.3,
        prev_variance=0.0,
        prev_count=1,
        value=0.5,
        alpha=DIMENSION_PRECISION_EWMA_ALPHA,
        min_variance=DIMENSION_PRECISION_MIN_VARIANCE["execution_pressure"],
    )
    assert state.dimension_precision_ewma_n["execution_pressure"] == 2
    assert state.dimension_precision_ewma["execution_pressure"] == expected.ewma
    assert state.dimension_precision_ewma_var["execution_pressure"] == expected.variance
    assert state.dimension_precision_zscore["execution_pressure"] == expected.zscore


def test_only_dimensions_present_this_tick_are_updated() -> None:
    """A tick that only produces a reading for one of the four channel-merge
    PRESSURE_DIMENSIONS must not fabricate baselines for the other three.

    2026-08-16: `deviation_pressure` is a 5th PRESSURE_DIMENSIONS entry, but
    it is NOT a channel-merge dimension -- it is a derived scalar
    (orion.field.pressure.field_pressures_with_provenance()) that is always
    present, at 0.0 on a quiet/untouched field, never absent. So it (and its
    baseline) is real and expected here even though this tick's
    node_vectors never mention it -- that is the honest "nothing admitted"
    reading, not the same "no channel mapped this tick" absence the other 3
    dimensions can have.
    """
    state = _empty_state("tick_0")
    state.node_vectors["node:athena"] = {"execution_pressure": 0.3}
    update_dimension_precision_baseline(state)
    pressures = field_pressures(state)
    assert set(pressures) == {"execution_pressure", "deviation_pressure"}
    assert state.dimension_precision_ewma_n == {"execution_pressure": 1, "deviation_pressure": 1}
    assert "resource_pressure" not in state.dimension_precision_ewma_n
    assert "reasoning_pressure" not in state.dimension_precision_ewma_n
    assert "reliability_pressure" not in state.dimension_precision_ewma_n


def test_upgrade_from_persisted_row_without_precision_fields_degrades_safely() -> None:
    """A FieldStateV1 persisted before this field existed loads with empty
    precision dicts (Pydantic default_factory=dict) -- the producer must
    treat that exactly like a genuine cold start, not raise."""
    payload = {
        "schema_version": "field.state.v1",
        "generated_at": BASE.isoformat(),
        "tick_id": "tick_pre_upgrade",
        "node_vectors": {"node:athena": {"execution_pressure": 0.4}},
    }
    state = FieldStateV1.model_validate(payload)
    assert state.dimension_precision_ewma_n == {}
    update_dimension_precision_baseline(state)
    assert state.dimension_precision_ewma_n["execution_pressure"] == 1
