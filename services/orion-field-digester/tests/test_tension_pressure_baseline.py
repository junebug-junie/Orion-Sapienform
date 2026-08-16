"""Producer-side deviation gate wiring (2026-08-16, docs/superpowers/specs/
2026-08-16-tension-driven-mutating-dispatch-design.md). Same shape as
test_dimension_precision_baseline.py -- a live digestion-tick producer must
carry state across restarts, unlike the offline measurement scripts that
rebuild a gate from scratch each run.
"""
from __future__ import annotations

from datetime import datetime, timezone

from orion.attention.tension import DeviationGate, FieldTensionCompetition, deviation_pressure
from orion.field.pressure import field_pressures
from orion.schemas.field_state import FieldStateV1

from app.digestion.tension import update_tension_pressure

BASE = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


def _empty_state(tick_id: str) -> FieldStateV1:
    return FieldStateV1(generated_at=BASE, tick_id=tick_id)


def test_first_tick_seeds_baseline_and_admits_nothing() -> None:
    state = _empty_state("tick_0")
    state.node_vectors["node:athena"] = {"cpu_pressure": 0.20}
    update_tension_pressure(state)
    assert state.tension_deviation_pressure == 0.0
    assert state.tension_baseline_n["node:athena\x1fcpu_pressure"] == 1
    assert state.tension_baseline_mu["node:athena\x1fcpu_pressure"] == 0.20


def test_deviation_pressure_is_zero_while_steady() -> None:
    state = _empty_state("tick_0")
    for i in range(8):
        state.tick_id = f"tick_{i}"
        state.node_vectors["node:athena"] = {"cpu_pressure": 0.20}
        update_tension_pressure(state)
    assert state.tension_deviation_pressure == 0.0


def test_a_real_step_change_after_warmup_raises_deviation_pressure() -> None:
    state = _empty_state("tick_0")
    for i in range(6):
        state.tick_id = f"tick_{i}"
        state.node_vectors["node:athena"] = {"cpu_pressure": 0.20}
        update_tension_pressure(state)
    assert state.tension_deviation_pressure == 0.0

    state.tick_id = "tick_spike"
    state.node_vectors["node:athena"] = {"cpu_pressure": 0.95}
    update_tension_pressure(state)
    assert state.tension_deviation_pressure > 0.0


def test_baseline_persisted_on_state_matches_a_continuous_in_process_run() -> None:
    """The real correctness bar: a producer restart (fresh FieldTensionCompetition
    rehydrated from persisted dicts each tick) must behave identically to one
    long-lived gate that was never torn down."""
    series = [0.10, 0.11, 0.09, 0.12, 0.10, 0.11, 0.60]

    state = _empty_state("tick_0")
    for i, value in enumerate(series):
        state.tick_id = f"tick_{i}"
        state.node_vectors["node:athena"] = {"cpu_pressure": value}
        update_tension_pressure(state)
    via_producer = state.tension_deviation_pressure

    continuous = FieldTensionCompetition(gate=DeviationGate())
    result = None
    for i, value in enumerate(series):
        result = continuous.observe_tick({"node_vectors": {"node:athena": {"cpu_pressure": value}}})
    via_continuous = deviation_pressure(result)

    assert via_producer == via_continuous


def test_upgrade_from_persisted_row_without_tension_fields_degrades_safely() -> None:
    """A FieldStateV1 persisted before these fields existed loads with empty
    tension dicts (Pydantic default_factory=dict) -- must behave as a genuine
    cold start, not raise."""
    payload = {
        "schema_version": "field.state.v1",
        "generated_at": BASE.isoformat(),
        "tick_id": "tick_pre_upgrade",
        "node_vectors": {"node:athena": {"cpu_pressure": 0.30}},
    }
    state = FieldStateV1.model_validate(payload)
    assert state.tension_baseline_mu == {}
    update_tension_pressure(state)
    assert state.tension_baseline_n["node:athena\x1fcpu_pressure"] == 1
    assert state.tension_deviation_pressure == 0.0


def test_malformed_flat_key_is_skipped_not_raised() -> None:
    """A key with no separator (hand-edited config, a future format change)
    must degrade this key to cold start, not crash the whole digestion tick."""
    state = _empty_state("tick_0")
    state.tension_baseline_mu = {"no_separator_here": 0.5}
    state.tension_baseline_var = {"no_separator_here": 0.0}
    state.tension_baseline_n = {"no_separator_here": 3}
    state.node_vectors["node:athena"] = {"cpu_pressure": 0.20}
    update_tension_pressure(state)  # must not raise
    assert "node:athena\x1fcpu_pressure" in state.tension_baseline_mu


def test_field_pressures_exposes_deviation_pressure_after_the_producer_runs() -> None:
    state = _empty_state("tick_0")
    for i in range(6):
        state.tick_id = f"tick_{i}"
        state.node_vectors["node:athena"] = {"cpu_pressure": 0.20}
        update_tension_pressure(state)
    state.tick_id = "tick_spike"
    state.node_vectors["node:athena"] = {"cpu_pressure": 0.95}
    update_tension_pressure(state)

    pressures = field_pressures(state)
    assert "deviation_pressure" in pressures
    assert pressures["deviation_pressure"] == state.tension_deviation_pressure


def test_direction_map_is_cached_not_reloaded_every_tick() -> None:
    """Code review finding, 2026-08-16: FieldTensionCompetition()'s default
    `directions` re-parsed config/attention/channel_direction_map.yaml from
    disk on every construction, and update_tension_pressure() constructed a
    fresh one every tick -- a real, avoidable disk read + YAML parse roughly
    every ~2s digestion tick, forever. Fixed with an lru_cache; this proves
    the cache actually returns the same object across calls rather than a
    merely-equal one."""
    from app.digestion.tension import _cached_direction_map

    first = _cached_direction_map()
    second = _cached_direction_map()
    assert first is second


def test_field_pressures_reports_deviation_pressure_even_on_an_empty_field() -> None:
    """0.0 is a real reading on an empty/quiet field, not an absent key -- the
    dimension must always be present so PRESSURE_DIMENSIONS scoring never
    silently degrades to the old absent-key 0.0 path for a different reason."""
    state = _empty_state("tick_0")
    assert field_pressures(state)["deviation_pressure"] == 0.0
