"""Level-aware significance: `loaded_steady`-scoped Borda competition over
real per-(node, channel) regime reads.

Every fixture is hand-computed and stated longhand -- see test_field_regime.py's
own header for why a tautological "assert regime == the constant just
imported" suite is worthless against a mutation harness.
"""
from __future__ import annotations

import pytest

from orion.field.significance import TickResult, compute_tick, sustained_load_pressure
from orion.field.regime import LOADED_LEVEL, MIN_REGIME_SAMPLES, STEADY_DISPERSION


def _field_json(node_channel_values: dict[tuple[str, str], float]) -> dict:
    """One tick's node_vectors payload from {(node_id, channel): value}."""
    node_vectors: dict[str, dict[str, float]] = {}
    for (node_id, channel), value in node_channel_values.items():
        node_vectors.setdefault(node_id, {})[channel] = value
    return {"node_vectors": node_vectors}


def _ticks(channel: str, node_id: str, values: list[float]) -> list[dict]:
    return [_field_json({(node_id, channel): v}) for v in values]


# Hand-verified: MIN_REGIME_SAMPLES=8, LOADED_LEVEL=0.70, STEADY_DISPERSION=0.02
# (orion/field/regime.py's own live-checked constants, imported not redefined).
assert MIN_REGIME_SAMPLES == 8
assert LOADED_LEVEL == 0.70
assert STEADY_DISPERSION == 0.02


def test_loaded_steady_channel_votes_and_wins() -> None:
    """8 samples alternating 0.81/0.815 -- median 0.8125, pstdev 0.0025.
    0.8125 >= 0.70 (loaded) and 0.0025 <= 0.02 (steady) -> loaded_steady,
    same fixture shape test_field_regime.py's own headline test uses."""
    values = [0.810, 0.815] * 4  # 8 samples
    ticks = _ticks("memory_pressure", "node:athena", values)

    tick = compute_tick(ticks, window_seconds=900.0)

    assert tick.channels_evaluated == 1
    assert tick.channels_loaded_steady == 1
    assert tick.loaded == {"memory_pressure": {"node:athena": pytest.approx(0.8125)}}
    assert tick.winner == "node:athena"
    assert sustained_load_pressure(tick) == pytest.approx(0.8125)


def test_calm_channel_does_not_vote() -> None:
    """Low and steady (0.02 flat) -- calm, not loaded_steady. Real
    memory_pressure-idle shape."""
    ticks = _ticks("memory_pressure", "node:athena", [0.02] * 10)

    tick = compute_tick(ticks, window_seconds=900.0)

    assert tick.channels_evaluated == 1
    assert tick.channels_loaded_steady == 0
    assert tick.loaded == {}
    assert tick.winner is None
    assert sustained_load_pressure(tick) == 0.0


def test_loaded_volatile_channel_does_not_vote() -> None:
    """High level (median 0.87) but high dispersion -- the independence-check
    exclusion this module's own docstring names explicitly. Values: eight
    samples alternating 0.20/1.0 give median 0.6 -- not loaded on its own, so
    use a clearly-loaded-but-swinging series instead: 0.75/0.99 alternating.
    median = 0.87, pstdev = 0.12 exactly (alternating +/- d about the mean
    gives d = (0.99-0.75)/2 = 0.12). 0.87 >= 0.70 (loaded) but
    0.12 > 0.02 (NOT steady) -> loaded_volatile, must not vote here."""
    values = [0.75, 0.99] * 4
    ticks = _ticks("power_pressure", "node:athena", values)

    tick = compute_tick(ticks, window_seconds=900.0)

    assert tick.channels_evaluated == 1
    assert tick.channels_loaded_steady == 0
    assert tick.loaded == {}
    assert sustained_load_pressure(tick) == 0.0


def test_fewer_than_min_regime_samples_does_not_vote() -> None:
    """7 samples, one short of MIN_REGIME_SAMPLES=8 -- excluded before regime
    is even computed, same value shape as the winning 8-sample case above so
    only the count differs."""
    values = [0.810, 0.815] * 3 + [0.810]  # 7 samples
    ticks = _ticks("memory_pressure", "node:athena", values)

    tick = compute_tick(ticks, window_seconds=900.0)

    assert tick.channels_evaluated == 0
    assert tick.channels_loaded_steady == 0
    assert tick.loaded == {}


def test_polarity_inverted_channel_reads_pressure_equivalent_not_raw_level() -> None:
    """`availability` is HIGHER_IS_BETTER. Alternating 0.199/0.201 (8 samples,
    not flat -- a perfectly flat series reads `static`/`no_new_input`, same
    reason test_field_regime.py's own fixtures alternate rather than repeat
    one value): median 0.200, pstdev (0.201-0.199)/2 = 0.001.
    pressure_equivalent_level = 1 - 0.200 = 0.800 (loaded), dispersion 0.001
    (steady) -> loaded_steady. Confirms this module reads channel_regime's
    polarity-corrected field, not raw level, for both the admission test and
    the ballot value."""
    ticks = _ticks("availability", "node:athena", [0.199, 0.201] * 4)

    tick = compute_tick(ticks, window_seconds=900.0)

    assert tick.channels_loaded_steady == 1
    assert tick.loaded == {"availability": {"node:athena": pytest.approx(0.8)}}
    assert sustained_load_pressure(tick) == pytest.approx(0.8)


def test_two_nodes_borda_ranks_by_pressure_equivalent_level() -> None:
    """Two channels, two nodes, each channel loaded_steady for both nodes but
    at different levels -- node:athena strictly higher on both, so it must
    win regardless of channel identity (same Borda "commensurable across
    scorers" property orion.attention.tension.competition already proves).
    Alternating pairs, not flat (see the polarity test's own note): each
    pair's median is its own mean, pstdev is half the pair's spread --
    memory_pressure athena/atlas medians 0.900/0.750 (pstdev 0.005 each),
    cpu_pressure athena/atlas medians 0.850/0.720 (pstdev 0.005 each), all
    four well under STEADY_DISPERSION=0.02 and at/above LOADED_LEVEL=0.70."""
    ticks = []
    for v_a, v_b in zip([0.895, 0.905] * 4, [0.745, 0.755] * 4):
        ticks.append(
            _field_json(
                {
                    ("node:athena", "memory_pressure"): v_a,
                    ("node:atlas", "memory_pressure"): v_b,
                }
            )
        )
    for v_a, v_b in zip([0.845, 0.855] * 4, [0.715, 0.725] * 4):
        ticks.append(
            _field_json(
                {
                    ("node:athena", "cpu_pressure"): v_a,
                    ("node:atlas", "cpu_pressure"): v_b,
                }
            )
        )

    tick = compute_tick(ticks, window_seconds=900.0)

    # channels_loaded_steady counts (node, channel) PAIRS, not distinct
    # channel names -- 2 channels x 2 nodes, all four loaded_steady.
    assert tick.channels_loaded_steady == 4
    assert tick.loaded == {
        "memory_pressure": {
            "node:athena": pytest.approx(0.90),
            "node:atlas": pytest.approx(0.75),
        },
        "cpu_pressure": {
            "node:athena": pytest.approx(0.85),
            "node:atlas": pytest.approx(0.72),
        },
    }
    assert tick.winner == "node:athena"
    assert sustained_load_pressure(tick) == pytest.approx(0.90)


def test_empty_window_returns_zero_and_no_winner() -> None:
    tick = compute_tick([], window_seconds=900.0)

    assert tick.channels_evaluated == 0
    assert tick.any_loaded is False
    assert tick.winner is None
    assert sustained_load_pressure(tick) == 0.0


def test_non_finite_and_non_numeric_values_are_skipped_not_crashed() -> None:
    """Mirrors iter_observations' own contract (non-finite/non-numeric values
    are skipped, not coerced) -- this module must not crash on malformed
    historical rows."""
    ticks = [
        {"node_vectors": {"node:athena": {"memory_pressure": "not-a-number"}}},
        {"node_vectors": {"node:athena": {"memory_pressure": float("nan")}}},
        {"node_vectors": "not-a-dict"},  # whole row malformed
    ] + _ticks("memory_pressure", "node:athena", [0.810, 0.815] * 4)

    tick = compute_tick(ticks, window_seconds=900.0)

    # Only the 8 well-formed trailing samples count.
    assert tick.channels_loaded_steady == 1
    assert tick.winner == "node:athena"
