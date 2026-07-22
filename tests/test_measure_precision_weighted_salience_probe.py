"""Deterministic unit tests for measure_precision_weighted_salience_probe.py.

No DB, no network. Everything here is pure: parsing, rolling-window replay (via the
real `precision_weighted_salience()` import, not a reimplementation), summarization,
and window selection all operate on plain synthetic data. Same sibling-module-by-
file-path loading pattern as test_measure_emergent_clustering_probe.py.

Lives in top-level `tests/`, not `scripts/analysis/tests/` (where this file originally
shipped in PR #1241) -- `scripts` is in `pyproject.toml`'s `norecursedirs`, so a bare
`pytest` run from repo root never discovered it there. Confirmed the same defect exists
for every sibling test under `scripts/analysis/tests/` (e.g.
`test_measure_emergent_clustering_probe.py`) -- out of scope to fix here, flagged
separately.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "measure_precision_weighted_salience_probe.py"
)
_spec = importlib.util.spec_from_file_location(
    "measure_precision_weighted_salience_probe", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_precision_weighted_salience_probe"] = mod
_spec.loader.exec_module(mod)

UTC = timezone.utc
BASE = datetime(2026, 7, 21, 0, 0, 0, tzinfo=UTC)


# ===========================================================================
# parse_prediction_error -- pure scalar parsing, must never raise.
# ===========================================================================


def test_parse_prediction_error_valid_string() -> None:
    assert mod.parse_prediction_error("0.0429") == 0.0429


def test_parse_prediction_error_none_returns_none() -> None:
    assert mod.parse_prediction_error(None) is None


def test_parse_prediction_error_malformed_returns_none() -> None:
    assert mod.parse_prediction_error("not-a-number") is None
    assert mod.parse_prediction_error("") is None


def test_parse_prediction_error_non_finite_returns_none() -> None:
    assert mod.parse_prediction_error("nan") is None
    assert mod.parse_prediction_error("inf") is None


# ===========================================================================
# compute_rolling_results -- exercises the REAL precision_weighted_salience()
# import, not a local copy.
# ===========================================================================


def test_compute_rolling_results_length_matches_input() -> None:
    values = [0.01, 0.02, 0.5, 0.03, 0.04]
    results = mod.compute_rolling_results(values, rolling_window=3)
    assert len(results) == len(values)


def test_compute_rolling_results_first_tick_uses_single_sample_window() -> None:
    values = [0.05, 0.1, 0.2]
    results = mod.compute_rolling_results(values, rolling_window=3)
    assert results[0].n_samples == 1
    assert results[0].current_error == 0.05


def test_compute_rolling_results_window_caps_at_rolling_size() -> None:
    values = [0.01] * 10 + [0.9]
    results = mod.compute_rolling_results(values, rolling_window=3)
    # last tick's window should be the 3 most recent samples: [0.01, 0.01, 0.9]
    assert results[-1].n_samples == 3
    assert results[-1].current_error == 0.9


# ===========================================================================
# summarize_results
# ===========================================================================


def test_summarize_results_empty_returns_none() -> None:
    assert mod.summarize_results([]) is None


def test_summarize_results_reports_variance_floored_rate() -> None:
    values = [0.02] * 10 + [0.5]  # last tick: real change against near-constant history
    results = mod.compute_rolling_results(values, rolling_window=20)
    summary = mod.summarize_results(results)
    assert summary is not None
    assert summary.n == 11
    assert summary.variance_floored_count >= 1
    assert 0.0 <= summary.variance_floored_pct <= 100.0


def test_summarize_results_healthy_series_no_floor_hits() -> None:
    values = [0.01, 0.08, 0.02, 0.12, 0.005, 0.09, 0.02, 0.15, 0.03, 0.11]
    results = mod.compute_rolling_results(values, rolling_window=20)
    summary = mod.summarize_results(results)
    assert summary is not None
    assert summary.raw_error_min == min(values)
    assert summary.raw_error_max == max(values)


# ===========================================================================
# choose_windows -- same anchoring/degradation contract as the clustering
# probe's own choose_windows, but with this script's much smaller
# MIN_WINDOW_HOURS default (real data volume here is retention-bounded to
# minutes, not days).
# ===========================================================================


def test_choose_windows_none_when_span_too_short() -> None:
    min_ts = BASE
    max_ts = BASE + timedelta(seconds=1)
    assert mod.choose_windows(min_ts, max_ts, window_hours=24, gap_hours=12) is None


def test_choose_windows_back_to_back_split_for_short_real_span() -> None:
    """Real Candidate A data (~30min retention window) never reaches the 24h/12h
    ask -- must degrade to a back-to-back split instead of returning None, as long
    as it clears this script's own MIN_WINDOW_HOURS floor (3 minutes)."""
    min_ts = BASE
    max_ts = BASE + timedelta(minutes=30)
    windows = mod.choose_windows(min_ts, max_ts, window_hours=24, gap_hours=12)
    assert windows is not None
    win_a, win_b = windows
    assert win_a.start == min_ts
    assert win_b.end == max_ts
    assert win_a.end == win_b.start  # back-to-back, no gap


def test_choose_windows_full_split_when_span_is_ample() -> None:
    min_ts = BASE
    max_ts = BASE + timedelta(hours=61)  # >= 2*24 + 12
    windows = mod.choose_windows(min_ts, max_ts, window_hours=24, gap_hours=12)
    assert windows is not None
    win_a, win_b = windows
    assert win_a.start == min_ts
    assert win_a.end == min_ts + timedelta(hours=24)
    assert win_b.end == max_ts
    assert win_b.start == max_ts - timedelta(hours=24)
    assert win_a.end < win_b.start  # real gap between the two windows


def test_choose_windows_none_min_ts_or_max_ts() -> None:
    assert mod.choose_windows(None, BASE, 24, 12) is None
    assert mod.choose_windows(BASE, None, 24, 12) is None


def test_choose_windows_none_when_max_before_min() -> None:
    assert mod.choose_windows(BASE + timedelta(hours=1), BASE, 24, 12) is None


# ===========================================================================
# partition_by_window
# ===========================================================================


def test_partition_by_window_no_row_double_counted_or_dropped() -> None:
    win_a = mod.WindowSpec(BASE, BASE + timedelta(minutes=5), "a")
    win_b = mod.WindowSpec(BASE + timedelta(minutes=10), BASE + timedelta(minutes=15), "b")
    boundary_ts = BASE + timedelta(minutes=5)  # == win_a.end
    timestamps = [BASE + timedelta(minutes=1), boundary_ts, BASE + timedelta(minutes=14)]
    values = [1.0, 2.0, 3.0]

    a, b = mod.partition_by_window(timestamps, values, win_a, win_b)

    assert a == [1.0]
    assert b == [3.0]  # boundary_ts falls in neither window (gap), row 3 in win_b
    assert len(a) + len(b) + 1 == len(values)  # the boundary row is dropped, not duplicated


def test_partition_by_window_last_row_inclusive_of_window_b_end() -> None:
    win_a = mod.WindowSpec(BASE, BASE + timedelta(minutes=2), "a")
    win_b = mod.WindowSpec(BASE + timedelta(minutes=8), BASE + timedelta(minutes=10), "b")
    timestamps = [BASE + timedelta(minutes=10)]  # exactly win_b.end
    values = [9.0]

    a, b = mod.partition_by_window(timestamps, values, win_a, win_b)

    assert a == []
    assert b == [9.0]
