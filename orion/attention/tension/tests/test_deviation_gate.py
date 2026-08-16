"""Gate behaviour. Fixtures are hand-computed, not read back off the implementation."""
from __future__ import annotations

import pytest

from orion.attention.tension.deviation_gate import DeviationGate


def _gate(**kw) -> DeviationGate:
    params = dict(alpha=0.1, z_threshold=1.5, relative_sigma_floor=0.01, warmup=5)
    params.update(kw)
    return DeviationGate(**params)


def _steady(gate: DeviationGate, n: int, x: float, worse: str = "up") -> None:
    for _ in range(n):
        gate.observe("node:a", "cpu_pressure", x, worse=worse)


def test_first_observation_seeds_and_admits_nothing():
    gate = _gate()
    assert gate.observe("node:a", "cpu_pressure", 0.9) == 0.0
    assert gate.baseline_count() == 1
    assert not gate.is_warm("node:a", "cpu_pressure")


def test_steady_input_never_admits():
    """The flood-starving property: a constant channel settles to its own mean."""
    gate = _gate()
    for _ in range(200):
        assert gate.observe("node:a", "cpu_pressure", 0.10) == 0.0


def test_warmup_suppresses_admission_but_still_trains():
    gate = _gate(warmup=5)
    for _ in range(3):
        gate.observe("node:a", "cpu_pressure", 0.10)
    assert gate.observe("node:a", "cpu_pressure", 0.99) == 0.0
    assert gate.is_warm("node:a", "cpu_pressure") is False


def test_step_change_admits_hand_computed_deviation():
    """10 observations at 0.10 -> mu=0.10, var=0.0 (delta is exactly 0 each fold).
    sigma = max(sqrt(0.0), 0.01 * |0.10|, 1e-12) = 0.001
    z = (0.50 - 0.10) / 0.001 = 400.0
    excess = (+1 * 400.0) - 1.5 = 398.5
    """
    gate = _gate()
    _steady(gate, 10, 0.10)
    assert gate.observe("node:a", "cpu_pressure", 0.50) == pytest.approx(398.5)


def test_wrong_direction_does_not_admit():
    """Same jump upward, but the channel is one where a FALL is worse:
    excess = (-1 * 400.0) - 1.5 < 0 -> no admission."""
    gate = _gate()
    _steady(gate, 10, 0.10, worse="down")
    assert gate.observe("node:a", "cpu_pressure", 0.50, worse="down") == 0.0


def test_fall_admits_on_a_down_is_worse_channel():
    """z = (0.02 - 0.10)/0.001 = -80.0; direction=-1 -> excess = 80.0 - 1.5 = 78.5"""
    gate = _gate()
    _steady(gate, 10, 0.10, worse="down")
    assert gate.observe("node:a", "cpu_pressure", 0.02, worse="down") == pytest.approx(78.5)


def test_confidence_scales_admission():
    gate = _gate()
    _steady(gate, 10, 0.10)
    assert gate.observe("node:a", "cpu_pressure", 0.50, confidence=0.5) == pytest.approx(199.25)


def test_baselines_are_independent_per_node_and_channel():
    gate = _gate()
    for _ in range(10):
        gate.observe("node:a", "cpu_pressure", 0.10)
    assert gate.observe("node:b", "cpu_pressure", 0.90) == 0.0
    assert gate.baseline_count() == 2


def test_sustained_deviation_habituates():
    """Adaptation is the rate-limiter: a step that stays put stops being news."""
    gate = _gate(alpha=0.3)
    _steady(gate, 10, 0.10)
    assert gate.observe("node:a", "cpu_pressure", 0.50) > 0.0
    last = 0.0
    for _ in range(120):
        last = gate.observe("node:a", "cpu_pressure", 0.50)
    assert last == 0.0, "a permanently-held value must stop admitting"


@pytest.mark.parametrize("bad", [None, "abc", float("nan"), float("inf")])
def test_bad_observation_degrades_to_zero_and_does_not_poison_baseline(bad):
    gate = _gate()
    _steady(gate, 10, 0.10)
    assert gate.observe("node:a", "cpu_pressure", bad) == 0.0
    # The baseline is untouched, so a real step still reads exactly as before.
    assert gate.observe("node:a", "cpu_pressure", 0.50) == pytest.approx(398.5)


# ---------------------------------------------------------------------------
# Regression: the absolute sigma_floor that broke scale-freedom (review, HIGH)
# ---------------------------------------------------------------------------


def test_small_variance_channel_can_still_admit():
    """Regression for the review's HIGH finding.

    Live case: `node:atlas` / `memory_pressure` over 10,528 ticks ranged
    0.1109-0.1171 with pstdev 8.8e-4 -- real, continuous variation -- and
    admitted 0 of 10,528 under the old absolute `sigma_floor=0.02`, because
    clearing it needed a ~0.03 absolute jump (a 26% relative move). It then
    appeared in the report's `never_admitted` list indistinguishable from a
    genuinely calm channel.

    Hand-computed with the relative floor: mu=0.1140, var=0 for a constant
    baseline, so sigma = 0.01 * 0.1140 = 0.001140.
    z = (0.1171 - 0.1140) / 0.001140 = 2.7193
    excess = 2.7193 - 1.5 = 1.2193
    """
    gate = _gate()
    for _ in range(10):
        gate.observe("node:atlas", "memory_pressure", 0.1140)
    assert gate.observe("node:atlas", "memory_pressure", 0.1171) == pytest.approx(
        1.2193, rel=1e-3
    )


def test_admission_is_invariant_under_rescaling_in_both_directions():
    """The floor is relative, so multiplying a channel by any k>0 scales mu,
    sigma and the floor identically and leaves the z-excess unchanged. The old
    absolute floor failed this for k<1 (and for any channel below its scale)."""
    baseline = [0.10, 0.20, 0.10, 0.20, 0.10, 0.20, 0.10, 0.20]

    def run(k: float) -> float:
        gate = _gate()
        for v in baseline:
            gate.observe("node:a", "cpu_pressure", v * k)
        return gate.observe("node:a", "cpu_pressure", 0.80 * k)

    at_1 = run(1.0)
    assert at_1 > 0.0, "test would be vacuous"
    assert run(10.0) == pytest.approx(at_1, rel=1e-9)
    assert run(0.1) == pytest.approx(at_1, rel=1e-9)
    assert run(1000.0) == pytest.approx(at_1, rel=1e-9)


# ---------------------------------------------------------------------------
# Regression: construction-parameter validation (review, MEDIUM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"alpha": -0.1}, "alpha"),
        ({"z_threshold": -1.0}, "z_threshold"),
        ({"relative_sigma_floor": -0.01}, "relative_sigma_floor"),
        ({"warmup": 0}, "warmup"),
        ({"warmup": -3}, "warmup"),
    ],
)
def test_invalid_construction_parameters_raise_immediately(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _gate(**kwargs)


def test_zero_relative_floor_does_not_divide_by_zero():
    """`relative_sigma_floor=0` is legal (it just disables the relative floor);
    the absolute _MIN_SIGMA backstop keeps it from dividing by zero on a
    constant baseline, which the old absolute `sigma_floor=0` did not."""
    gate = _gate(relative_sigma_floor=0.0)
    _steady(gate, 10, 0.10)
    assert gate.observe("node:a", "cpu_pressure", 0.50) > 0.0


def test_zero_mean_baseline_does_not_divide_by_zero():
    gate = _gate()
    _steady(gate, 10, 0.0)
    assert gate.observe("node:a", "cpu_pressure", 0.5) > 0.0


# --- persistence (2026-08-16, tension-driven-mutating-dispatch) -------------
#
# A live producer must carry baselines across restarts -- the offline
# measurement scripts rebuild a gate from scratch each run, which only works
# for a one-shot analysis. export_baselines/import_baselines are the whole
# persistence contract; these tests are the round-trip proof.


def test_export_baselines_reflects_real_observations():
    gate = _gate()
    _steady(gate, 6, 0.20)
    exported = gate.export_baselines()
    mu, var, count = exported[("node:a", "cpu_pressure")]
    assert count == 6
    assert mu == pytest.approx(0.20)
    assert var == pytest.approx(0.0)  # constant input never accrues variance


def test_import_baselines_round_trips_admission_behavior():
    """A gate rehydrated from an export must behave identically to the
    original, not just carry the same numbers -- the real test is whether it
    admits the same things."""
    source = _gate()
    _steady(source, 10, 0.10)
    exported = source.export_baselines()

    rehydrated = _gate()
    rehydrated.import_baselines(exported)

    x = 0.50
    assert rehydrated.observe("node:a", "cpu_pressure", x) == pytest.approx(
        source.observe("node:a", "cpu_pressure", x)
    )


def test_import_baselines_replaces_not_merges():
    gate = _gate()
    _steady(gate, 6, 0.10)
    assert gate.baseline_count() == 1
    gate.import_baselines({("node:b", "memory_pressure"): (0.30, 0.0, 6)})
    assert gate.baseline_count() == 1
    assert ("node:a", "cpu_pressure") not in gate.export_baselines()


def test_export_on_a_fresh_gate_is_empty():
    assert _gate().export_baselines() == {}
