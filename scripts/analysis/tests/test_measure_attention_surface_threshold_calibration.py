"""Deterministic unit tests for measure_attention_surface_threshold_calibration.py.

No DB, no network. Everything exercised here is pure: percentile math,
concentration/pseudo-replication accounting, latest-per-theme extraction,
candidate-threshold evaluation, the daily percentile-stability replay, and
the age-gate check all operate on plain synthetic `TraceRow` lists. Same
sibling-module-by-file-path loading pattern as
test_measure_capability_channel_health.py.

Covers the edge cases the chosen repair strategy actually has to reason
about even though production code ended up as a static constant, not a
live percentile rule: very few rows, all-identical scores, and a sudden
distribution shift -- because this script's own percentile-stability replay
is the evidence base the PR cites for rejecting a percentile-based rule at
the current data volume, and that evidence has to be trustworthy on its own
degenerate inputs before it can be trusted on real ones.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_attention_surface_threshold_calibration.py"
_spec = importlib.util.spec_from_file_location("measure_attention_surface_threshold_calibration", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_attention_surface_threshold_calibration"] = mod
_spec.loader.exec_module(mod)


def _row(theme_key: str, salience: float, created_at: datetime) -> "mod.TraceRow":
    return mod.TraceRow(created_at=created_at, theme_key=theme_key, salience=salience)


_NOW = datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc)


# ===========================================================================
# compute_percentile -- pure math, must match Postgres percentile_cont shape.
# ===========================================================================


def test_compute_percentile_single_value_returns_that_value() -> None:
    assert mod.compute_percentile([0.42], 50) == 0.42
    assert mod.compute_percentile([0.42], 95) == 0.42


def test_compute_percentile_empty_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        mod.compute_percentile([], 50)


def test_compute_percentile_all_identical_values() -> None:
    values = [0.4] * 50
    assert mod.compute_percentile(values, 50) == 0.4
    assert mod.compute_percentile(values, 95) == 0.4


def test_compute_percentile_known_interpolation() -> None:
    values = [0.0, 1.0]
    assert mod.compute_percentile(values, 50) == 0.5
    assert mod.compute_percentile(values, 0) == 0.0
    assert mod.compute_percentile(values, 100) == 1.0


# ===========================================================================
# compute_distribution_stats -- degenerate + normal inputs.
# ===========================================================================


def test_compute_distribution_stats_empty_is_not_populated() -> None:
    stats = mod.compute_distribution_stats([])
    assert stats.count == 0
    assert not stats.is_populated
    assert stats.min_v is None


def test_compute_distribution_stats_all_identical_scores_has_zero_stdev() -> None:
    rows = [_row("t1", 0.4, _NOW) for _ in range(10)]
    stats = mod.compute_distribution_stats(rows)
    assert stats.count == 10
    assert stats.min_v == stats.max_v == 0.4
    assert stats.stdev_v == 0.0
    assert all(v == 0.4 for v in stats.percentiles.values())


def test_compute_distribution_stats_very_few_rows() -> None:
    rows = [_row("t1", 0.3, _NOW), _row("t2", 0.5, _NOW)]
    stats = mod.compute_distribution_stats(rows)
    assert stats.count == 2
    assert stats.min_v == 0.3
    assert stats.max_v == 0.5


# ===========================================================================
# count_rows_at_or_above -- the "is the current gate reachable" check.
# ===========================================================================


def test_count_rows_at_or_above_reproduces_the_confirmed_live_bug() -> None:
    # Real shape confirmed live 2026-07-31: max ever recorded is 0.490.
    rows = [_row("t1", v, _NOW) for v in (0.35, 0.41, 0.458, 0.489, 0.490)]
    assert mod.count_rows_at_or_above(rows, 0.5) == 0
    assert mod.count_rows_at_or_above(rows, 0.40) == 4


# ===========================================================================
# compute_concentration -- pseudo-replication check.
# ===========================================================================


def test_compute_concentration_single_theme_is_fully_concentrated() -> None:
    rows = [_row("only-theme", 0.4, _NOW) for _ in range(20)]
    result = mod.compute_concentration(rows)
    assert result.distinct_theme_count == 1
    assert result.top2_share == 1.0


def test_compute_concentration_matches_measured_live_shape() -> None:
    # Confirmed live 2026-07-31: top 2 of 6 theme_keys account for 89.8% of
    # all rows. Reproduced here with an equivalent skew, not the exact
    # counts, to test the accounting logic rather than pin a live number.
    rows = (
        [_row("loud-1", 0.4, _NOW) for _ in range(150)]
        + [_row("loud-2", 0.4, _NOW) for _ in range(140)]
        + [_row("quiet-1", 0.4, _NOW) for _ in range(6)]
        + [_row("quiet-2", 0.4, _NOW) for _ in range(12)]
        + [_row("quiet-3", 0.4, _NOW) for _ in range(20)]
        + [_row("quiet-4", 0.4, _NOW) for _ in range(4)]
    )
    result = mod.compute_concentration(rows)
    assert result.distinct_theme_count == 6
    assert result.top2_share is not None
    assert result.top2_share > 0.85


def test_compute_concentration_empty() -> None:
    result = mod.compute_concentration([])
    assert result.distinct_theme_count == 0
    assert result.top2_share is None


# ===========================================================================
# compute_latest_per_theme -- the exact population load_pending_loops()
# actually gates against (most recent row per theme, not historical max).
# ===========================================================================


def test_compute_latest_per_theme_picks_most_recent_not_max() -> None:
    rows = [
        _row("t1", 0.9, _NOW - timedelta(days=5)),  # old peak, not "latest"
        _row("t1", 0.3, _NOW),  # latest -- this is what should be reported
        _row("t2", 0.5, _NOW - timedelta(hours=1)),
    ]
    latest = mod.compute_latest_per_theme(rows)
    by_theme = {x.theme_key: x.salience for x in latest}
    assert by_theme["t1"] == 0.3
    assert by_theme["t2"] == 0.5


def test_compute_latest_per_theme_sorted_descending_by_salience() -> None:
    rows = [_row("low", 0.2, _NOW), _row("high", 0.8, _NOW), _row("mid", 0.5, _NOW)]
    latest = mod.compute_latest_per_theme(rows)
    assert [x.theme_key for x in latest] == ["high", "mid", "low"]


def test_compute_latest_per_theme_empty() -> None:
    assert mod.compute_latest_per_theme([]) == []


# ===========================================================================
# evaluate_candidate_thresholds
# ===========================================================================


def test_evaluate_candidate_thresholds_matches_measured_live_result() -> None:
    # Real per-theme latest scores confirmed live 2026-07-31.
    rows = [
        _row("open-loop-e11d318a9036", 0.458, _NOW),
        _row("open-loop-3be13c7644a0", 0.413, _NOW),
        _row("open-loop-8f8766373aba", 0.385, _NOW),
        _row("open-loop-1c822db7221d", 0.384, _NOW),
        _row("open-loop-733aad95961b", 0.366, _NOW),
        _row("open-loop-9d84d08cddf5", 0.356, _NOW),
    ]
    latest = mod.compute_latest_per_theme(rows)
    results = mod.evaluate_candidate_thresholds(latest, (0.35, 0.40, 0.45, 0.50))
    by_threshold = {r.threshold: r.surfaced_count for r in results}
    assert by_threshold[0.35] == 6
    assert by_threshold[0.40] == 2
    assert by_threshold[0.45] == 1
    assert by_threshold[0.50] == 0


# ===========================================================================
# compute_daily_percentile_stability -- the small-n instability + sudden
# distribution shift edge cases.
# ===========================================================================


def test_daily_percentile_stability_empty() -> None:
    assert mod.compute_daily_percentile_stability([]) == []


def test_daily_percentile_stability_very_few_rows_single_day() -> None:
    rows = [_row("t1", 0.4, _NOW), _row("t2", 0.5, _NOW)]
    points = mod.compute_daily_percentile_stability(rows)
    assert len(points) == 1
    assert points[0].cumulative_n == 2


def test_daily_percentile_stability_detects_sudden_distribution_shift() -> None:
    # First 3 days: low, stable scores (small n). Day 4: a sudden jump to
    # much higher scores. A percentile-based rule computed on day 3 would
    # have set a bar the day-4 population blows past -- exactly the
    # "shifts confusingly even when a loop's own score hasn't changed"
    # risk named in the task. This test proves the replay actually
    # detects that shift rather than smoothing over it.
    day0 = _NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    for d in range(3):
        for _ in range(10):
            rows.append(_row(f"stable-{d}", 0.30, day0 + timedelta(days=d)))
    for _ in range(10):
        rows.append(_row("shifted", 0.90, day0 + timedelta(days=3)))
    points = mod.compute_daily_percentile_stability(rows, pct=85.0)
    assert len(points) == 4
    pre_shift = points[2].cumulative_p85
    post_shift = points[3].cumulative_p85
    assert pre_shift is not None and post_shift is not None
    assert post_shift > pre_shift
    # trailing-7d window includes the shift immediately (short lookback)
    assert points[3].trailing7_p85 == post_shift


def test_daily_percentile_stability_all_identical_scores_is_flat() -> None:
    day0 = _NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = [_row("t1", 0.4, day0 + timedelta(days=d)) for d in range(5)]
    points = mod.compute_daily_percentile_stability(rows, pct=85.0)
    values = [p.cumulative_p85 for p in points if p.cumulative_p85 is not None]
    assert values and all(v == 0.4 for v in values)


# ===========================================================================
# compute_age_gate_results -- confirms SURFACE_MIN_AGE_SEC binding check.
# ===========================================================================


def test_age_gate_results_new_theme_is_blocked() -> None:
    rows = [_row("new", 0.9, _NOW - timedelta(seconds=10))]
    results = mod.compute_age_gate_results(rows, _NOW, min_age_sec=300.0)
    assert len(results) == 1
    assert results[0].clears_current_gate is False


def test_age_gate_results_old_theme_clears() -> None:
    rows = [_row("old", 0.9, _NOW - timedelta(days=6))]
    results = mod.compute_age_gate_results(rows, _NOW, min_age_sec=300.0)
    assert results[0].clears_current_gate is True


def test_age_gate_results_uses_earliest_row_per_theme() -> None:
    rows = [
        _row("t1", 0.9, _NOW - timedelta(days=6)),  # earliest
        _row("t1", 0.9, _NOW - timedelta(seconds=5)),  # most recent, ignored for age
    ]
    results = mod.compute_age_gate_results(rows, _NOW, min_age_sec=300.0)
    assert len(results) == 1
    assert results[0].age_seconds >= timedelta(days=6).total_seconds() - 1
    assert results[0].clears_current_gate is True
