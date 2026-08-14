"""Gate behaviour. Fixtures are hand-computed, not read back off the implementation."""
from __future__ import annotations

import pytest

from orion.attention.tension.deviation_gate import DeviationGate


def _steady(gate: DeviationGate, n: int, x: float, worse: str = "up") -> None:
    for _ in range(n):
        gate.observe("node:a", "cpu_pressure", x, worse=worse)


def test_first_observation_seeds_and_admits_nothing():
    gate = DeviationGate()
    assert gate.observe("node:a", "cpu_pressure", 0.9) == 0.0
    assert gate.baseline_count() == 1
    assert not gate.is_warm("node:a", "cpu_pressure")


def test_steady_input_never_admits():
    """The flood-starving property: a constant channel settles to its own mean."""
    gate = DeviationGate()
    for _ in range(200):
        assert gate.observe("node:a", "cpu_pressure", 0.10) == 0.0


def test_warmup_suppresses_admission_but_still_trains():
    gate = DeviationGate(warmup=5)
    # 4 observations total -> count==4 < warmup, so even a large jump admits nothing.
    for _ in range(3):
        gate.observe("node:a", "cpu_pressure", 0.10)
    assert gate.observe("node:a", "cpu_pressure", 0.99) == 0.0
    assert gate.is_warm("node:a", "cpu_pressure") is False


def test_step_change_admits_hand_computed_deviation():
    """10 observations at 0.10 -> mu=0.10, var=0.0 (delta is exactly 0 each fold).
    sigma = max(sqrt(0.0), sigma_floor) = 0.02.
    z = (0.50 - 0.10) / 0.02 = 20.0
    excess = (+1 * 20.0) - 1.5 = 18.5
    """
    gate = DeviationGate(alpha=0.1, z_threshold=1.5, sigma_floor=0.02, warmup=5)
    _steady(gate, 10, 0.10)
    assert gate.observe("node:a", "cpu_pressure", 0.50) == pytest.approx(18.5)


def test_wrong_direction_does_not_admit():
    """Same 20-sigma jump upward, but the channel is one where a FALL is worse.
    direction = -1, so excess = (-1 * 20.0) - 1.5 = -21.5 -> no admission.
    """
    gate = DeviationGate(alpha=0.1, z_threshold=1.5, sigma_floor=0.02, warmup=5)
    _steady(gate, 10, 0.10, worse="down")
    assert gate.observe("node:a", "cpu_pressure", 0.50, worse="down") == 0.0


def test_fall_admits_on_a_down_is_worse_channel():
    """z = (0.02 - 0.10)/0.02 = -4.0; direction=-1 -> excess = 4.0 - 1.5 = 2.5"""
    gate = DeviationGate(alpha=0.1, z_threshold=1.5, sigma_floor=0.02, warmup=5)
    _steady(gate, 10, 0.10, worse="down")
    assert gate.observe("node:a", "cpu_pressure", 0.02, worse="down") == pytest.approx(2.5)


def test_confidence_scales_admission():
    gate = DeviationGate(alpha=0.1, z_threshold=1.5, sigma_floor=0.02, warmup=5)
    _steady(gate, 10, 0.10)
    assert gate.observe("node:a", "cpu_pressure", 0.50, confidence=0.5) == pytest.approx(9.25)


def test_baselines_are_independent_per_node_and_channel():
    gate = DeviationGate(warmup=5)
    for _ in range(10):
        gate.observe("node:a", "cpu_pressure", 0.10)
    # A different node, same channel, is a cold baseline -- it must not inherit.
    assert gate.observe("node:b", "cpu_pressure", 0.90) == 0.0
    assert gate.baseline_count() == 2


def test_sustained_deviation_habituates():
    """Adaptation is the rate-limiter: a step that stays put stops being news."""
    gate = DeviationGate(alpha=0.3, z_threshold=1.5, sigma_floor=0.02, warmup=5)
    _steady(gate, 10, 0.10)
    first = gate.observe("node:a", "cpu_pressure", 0.50)
    assert first > 0.0
    last = first
    for _ in range(60):
        last = gate.observe("node:a", "cpu_pressure", 0.50)
    assert last == 0.0, "a permanently-held value must stop admitting"


@pytest.mark.parametrize("bad", [None, "abc", float("nan"), float("inf")])
def test_bad_input_degrades_to_zero_and_does_not_poison_baseline(bad):
    gate = DeviationGate(warmup=5)
    _steady(gate, 10, 0.10)
    assert gate.observe("node:a", "cpu_pressure", bad) == 0.0
    # The baseline is untouched, so a real step still reads exactly as before.
    assert gate.observe("node:a", "cpu_pressure", 0.50) == pytest.approx(18.5)
