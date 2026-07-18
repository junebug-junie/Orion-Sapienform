"""Deterministic unit tests for measure_self_state_signal_quality.py.

No DB. All pure functions operate on synthetic float series / DimensionSample
lists, same module-loading pattern as
scripts/analysis/tests/test_measure_origination_gate.py.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_self_state_signal_quality.py"
_spec = importlib.util.spec_from_file_location("measure_self_state_signal_quality", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_self_state_signal_quality"] = mod
_spec.loader.exec_module(mod)

UTC = timezone.utc
BASE = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)


def test_rolling_std_zero_for_constant_series() -> None:
    stds = mod.rolling_std([0.5] * 30, window=10)
    assert all(s == 0.0 for s in stds)


def test_rolling_std_nonzero_for_varying_series() -> None:
    stds = mod.rolling_std([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], window=3)
    assert stds[-1] > 0.0


def test_linear_drift_per_hour_detects_upward_slope() -> None:
    # +1.0 per hour, sampled every 600s (10 min)
    times = [i * 600.0 for i in range(7)]
    values = [t / 3600.0 for t in times]
    drift = mod.linear_drift_per_hour(times, values)
    assert drift is not None
    assert drift == pytest.approx(1.0)


def test_linear_drift_per_hour_none_for_single_point() -> None:
    assert mod.linear_drift_per_hour([0.0], [0.5]) is None


def test_zero_crossing_period_estimate_flat_series_has_no_crossings() -> None:
    period, crossings = mod.zero_crossing_period_estimate([0.5] * 20)
    assert crossings == 0
    assert period is None


def test_zero_crossing_period_estimate_detects_fast_alternation() -> None:
    series = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    period, crossings = mod.zero_crossing_period_estimate(series)
    assert crossings >= 2
    assert period is not None
    assert period <= 3.0


def test_diagnose_dimension_flags_pinned_flat_series() -> None:
    samples = [
        mod.DimensionSample(generated_at=BASE + timedelta(seconds=i), score=0.5, confidence=0.9)
        for i in range(30)
    ]
    diag = mod.diagnose_dimension("coherence", samples)
    assert diag.n == 30
    assert "pinned_or_flat" in diag.flags


def test_diagnose_dimension_flags_fast_oscillation() -> None:
    samples = [
        mod.DimensionSample(
            generated_at=BASE + timedelta(seconds=i), score=(0.9 if i % 2 == 0 else 0.1), confidence=0.9
        )
        for i in range(40)
    ]
    diag = mod.diagnose_dimension("uncertainty", samples)
    assert "fast_oscillation_sawtooth_suspect" in diag.flags


def test_diagnose_dimension_no_data_flag_for_empty_series() -> None:
    diag = mod.diagnose_dimension("social_pressure", [])
    assert diag.n == 0
    assert diag.flags == ["no_data"]


def test_diagnose_dimension_nominal_for_healthy_slow_varying_series() -> None:
    # A real, non-trivial sine (amplitude 0.1, period 60 ticks) sampled over
    # ~3.3 periods: has real noise floor (not flat), a slow oscillation
    # period well above the fast-oscillation threshold, and near-zero net
    # drift over the full window -- should trip none of the three
    # heuristics.
    samples = []
    for i in range(200):
        score = 0.5 + 0.1 * math.sin(2 * math.pi * i / 60.0)
        samples.append(
            mod.DimensionSample(generated_at=BASE + timedelta(minutes=i), score=score, confidence=0.9)
        )
    diag = mod.diagnose_dimension("execution_pressure", samples)
    assert diag.flags == ["nominal"]
