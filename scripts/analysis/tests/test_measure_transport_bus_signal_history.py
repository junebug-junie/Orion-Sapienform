"""Deterministic unit tests for measure_transport_bus_signal_history.py.

No DB. All pure functions operate on synthetic BusTick lists, same
module-loading pattern as scripts/analysis/tests/test_measure_self_state_signal_quality.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_transport_bus_signal_history.py"
_spec = importlib.util.spec_from_file_location("measure_transport_bus_signal_history", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_transport_bus_signal_history"] = mod
_spec.loader.exec_module(mod)

UTC = timezone.utc
BASE = datetime(2026, 7, 22, 0, 0, 0, tzinfo=UTC)


def _tick(i: int, *, depth: int = 91, counts=None, pressures=None) -> "mod.BusTick":
    return mod.BusTick(
        observed_at=BASE + timedelta(seconds=i * 10),
        max_stream_depth=depth,
        counts=counts or {},
        pressures=pressures or {},
    )


def test_compute_baseline_empty_returns_n_zero() -> None:
    baseline = mod.compute_baseline([])
    assert baseline.n == 0
    assert baseline.earliest is None
    assert all(v == 0 for v in baseline.count_totals.values())
    assert all(v is False for v in baseline.pressure_ever_nonzero.values())


def test_compute_baseline_flat_series_reports_zero_stdev_and_no_incidents() -> None:
    ticks = [_tick(i, depth=91) for i in range(10)]
    baseline = mod.compute_baseline(ticks)
    assert baseline.n == 10
    assert baseline.depth_min == 91
    assert baseline.depth_max == 91
    assert baseline.depth_stdev == 0.0
    assert all(v is False for v in baseline.pressure_ever_nonzero.values())
    assert all(v == 0 for v in baseline.count_totals.values())


def test_compute_baseline_detects_varying_depth() -> None:
    depths = [50, 100, 150, 200, 90]
    ticks = [_tick(i, depth=d) for i, d in enumerate(depths)]
    baseline = mod.compute_baseline(ticks)
    assert baseline.depth_min == 50
    assert baseline.depth_max == 200
    assert baseline.depth_stdev > 0.0
    assert baseline.depth_p50 is not None


def test_compute_baseline_tracks_first_nonzero_count() -> None:
    ticks = [
        _tick(0, counts={"backpressure_count": 0}),
        _tick(1, counts={"backpressure_count": 0}),
        _tick(2, counts={"backpressure_count": 3}),
        _tick(3, counts={"backpressure_count": 2}),
    ]
    baseline = mod.compute_baseline(ticks)
    assert baseline.count_totals["backpressure_count"] == 5
    assert baseline.count_first_nonzero_at["backpressure_count"] == ticks[2].observed_at.isoformat()
    # untouched counters stay at their default
    assert baseline.count_totals["observer_failure_count"] == 0
    assert baseline.count_first_nonzero_at["observer_failure_count"] is None


def test_compute_baseline_pressure_ever_nonzero_flag() -> None:
    ticks = [
        _tick(0, pressures={"contract_pressure": 0.0}),
        _tick(1, pressures={"contract_pressure": 0.4}),
    ]
    baseline = mod.compute_baseline(ticks)
    assert baseline.pressure_ever_nonzero["contract_pressure"] is True
    assert baseline.pressure_ever_nonzero["backpressure"] is False


def test_percentile_single_value() -> None:
    assert mod._percentile([5.0], 0.99) == 5.0


def test_percentile_empty_returns_none() -> None:
    assert mod._percentile([], 0.5) is None


def test_percentile_p50_of_sorted_list() -> None:
    values = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mod._percentile(values, 0.5) == 3.0


def test_render_report_insufficient_data_when_no_ticks() -> None:
    baseline = mod.compute_baseline([])
    report = mod.render_report(window_hours=24.0, baseline=baseline, truncated=False)
    assert "INSUFFICIENT REAL DATA" in report


def test_render_report_flags_all_flat_finding() -> None:
    ticks = [_tick(i, depth=91) for i in range(5)]
    baseline = mod.compute_baseline(ticks)
    report = mod.render_report(window_hours=1.0, baseline=baseline, truncated=False)
    assert "Every signal read as flat/zero" in report


def test_render_report_does_not_flag_flat_finding_when_a_signal_fired() -> None:
    ticks = [
        _tick(0, depth=91),
        _tick(1, depth=91, pressures={"backpressure": 0.5}),
    ]
    baseline = mod.compute_baseline(ticks)
    report = mod.render_report(window_hours=1.0, baseline=baseline, truncated=False)
    assert "Every signal read as flat/zero" not in report


def _tick_at(seconds_from_base: float) -> "mod.BusTick":
    return mod.BusTick(
        observed_at=BASE + timedelta(seconds=seconds_from_base),
        max_stream_depth=91,
    )


def test_compute_cadence_stats_empty_for_fewer_than_two_ticks() -> None:
    stats = mod.compute_cadence_stats([_tick_at(0)])
    assert stats.n_gaps == 0
    assert stats.median_gap_sec is None
    assert stats.stall_count == 0


def test_compute_cadence_stats_regular_cadence_no_stalls() -> None:
    ticks = [_tick_at(i * 10.0) for i in range(20)]
    stats = mod.compute_cadence_stats(ticks)
    assert stats.n_gaps == 19
    assert stats.median_gap_sec == pytest.approx(10.0)
    assert stats.stall_count == 0


def test_compute_cadence_stats_detects_a_real_stall() -> None:
    # Regular 10s cadence, then one 200s gap (a real reducer stall), then regular again.
    ticks = [_tick_at(i * 10.0) for i in range(10)]
    last = ticks[-1].observed_at
    ticks.append(mod.BusTick(observed_at=last + timedelta(seconds=200.0), max_stream_depth=91))
    ticks.extend(
        mod.BusTick(observed_at=ticks[-1].observed_at + timedelta(seconds=i * 10.0), max_stream_depth=91)
        for i in range(1, 5)
    )
    stats = mod.compute_cadence_stats(ticks)
    assert stats.stall_count == 1
    assert stats.max_gap_sec == pytest.approx(200.0)
    assert stats.worst_stall_at is not None
