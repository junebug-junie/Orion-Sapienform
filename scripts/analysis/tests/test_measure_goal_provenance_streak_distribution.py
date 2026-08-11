"""Deterministic unit tests for measure_goal_provenance_streak_distribution.py.

No DB, no network. Everything here is pure: run reconstruction, length
distribution, and qualification-rate computation all operate on plain
synthetic StreakTickRow sequences. Same sibling-module-by-file-path loading
pattern as test_measure_emergent_clustering_probe.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_goal_provenance_streak_distribution.py"
_spec = importlib.util.spec_from_file_location("measure_goal_provenance_streak_distribution", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_goal_provenance_streak_distribution"] = mod
_spec.loader.exec_module(mod)

UTC = timezone.utc
BASE = datetime(2026, 8, 11, 0, 0, 0, tzinfo=UTC)


def _row(offset_sec: int, target_id, count: int, *, min_streak: int = 3, qualified: bool | None = None):
    if qualified is None:
        qualified = count >= min_streak
    return mod.StreakTickRow(
        observed_at=BASE + timedelta(seconds=offset_sec),
        target_id=target_id,
        streak_count=count,
        min_streak_at_tick=min_streak,
        qualified=qualified,
    )


# ===========================================================================
# reconstruct_streak_runs
# ===========================================================================


def test_reconstruct_streak_runs_single_completed_run() -> None:
    rows = [
        _row(0, "node:a", 1),
        _row(1, "node:a", 2),
        _row(2, "node:a", 3),
        _row(3, "node:b", 1),  # a's run ends here, b's starts
    ]
    runs = mod.reconstruct_streak_runs(rows)
    assert len(runs) == 2
    a_run, b_run = runs
    assert a_run.target_id == "node:a"
    assert a_run.max_count == 3
    assert a_run.n_ticks == 3
    assert a_run.ongoing is False
    assert a_run.qualified is True  # crossed min_streak=3
    assert b_run.target_id == "node:b"
    assert b_run.ongoing is True  # last row in the input -- still live


def test_reconstruct_streak_runs_none_target_id_ends_run_without_producing_one() -> None:
    """A tick with no winner (update_dominance_streak's reset case) ends the current
    run but is not itself a zero-length run -- rows with target_id=None are dropped."""
    rows = [
        _row(0, "node:a", 1),
        _row(1, "node:a", 2),
        _row(2, None, 0),
        _row(3, "node:b", 1),
    ]
    runs = mod.reconstruct_streak_runs(rows)
    assert [r.target_id for r in runs] == ["node:a", "node:b"]
    assert runs[0].max_count == 2
    assert runs[0].ongoing is False


def test_reconstruct_streak_runs_never_qualified_run() -> None:
    rows = [_row(0, "node:a", 1, min_streak=3), _row(1, "node:a", 2, min_streak=3), _row(2, "node:b", 1)]
    runs = mod.reconstruct_streak_runs(rows)
    a_run = runs[0]
    assert a_run.max_count == 2
    assert a_run.qualified is False  # never reached min_streak=3


def test_reconstruct_streak_runs_empty_input() -> None:
    assert mod.reconstruct_streak_runs([]) == []


def test_reconstruct_streak_runs_all_none_target_ids() -> None:
    rows = [_row(0, None, 0), _row(1, None, 0)]
    assert mod.reconstruct_streak_runs(rows) == []


# ===========================================================================
# length_distribution / qualification_rate_at / target_id_distribution
# ===========================================================================


def test_length_distribution_excludes_ongoing_run() -> None:
    rows = [
        _row(0, "node:a", 1),
        _row(1, "node:a", 2),
        _row(2, "node:b", 1),  # a completed at length 2; b is ongoing
    ]
    runs = mod.reconstruct_streak_runs(rows)
    dist = mod.length_distribution(runs)
    assert dist == {2: 1}  # only a's completed run counted


def test_qualification_rate_at_computes_real_survival_rate() -> None:
    # Three completed runs of length 1, 2, 3 (target changes each time), plus a trailing
    # ongoing run that must not affect the rate.
    rows = [
        _row(0, "node:a", 1, min_streak=3),
        _row(1, "node:b", 1, min_streak=3),
        _row(2, "node:b", 2, min_streak=3),
        _row(3, "node:c", 1, min_streak=3),
        _row(4, "node:c", 2, min_streak=3),
        _row(5, "node:c", 3, min_streak=3),
        _row(6, "node:d", 1, min_streak=3),  # ongoing, excluded
    ]
    runs = mod.reconstruct_streak_runs(rows)
    # Completed runs: a=1, b=2, c=3 -- only c clears min_streak=3.
    assert mod.qualification_rate_at(runs, 3) == 1 / 3
    assert mod.qualification_rate_at(runs, 1) == 3 / 3
    assert mod.qualification_rate_at(runs, 4) == 0.0


def test_qualification_rate_at_returns_none_with_no_completed_runs() -> None:
    rows = [_row(0, "node:a", 1)]  # single run, still ongoing
    runs = mod.reconstruct_streak_runs(rows)
    assert mod.qualification_rate_at(runs, 3) is None


def test_reconstruct_streak_runs_marks_first_run_left_censored_when_requested() -> None:
    rows = [
        _row(0, "node:a", 5),  # would be mid-stream in real data if fetched from a truncated window
        _row(1, "node:b", 1),  # a's run ends; b is ongoing
    ]
    runs = mod.reconstruct_streak_runs(rows, first_run_left_censored=True)
    assert runs[0].target_id == "node:a"
    assert runs[0].left_censored is True
    assert runs[1].left_censored is False  # only the first run gets marked, not every run


def test_reconstruct_streak_runs_left_censored_default_false() -> None:
    rows = [_row(0, "node:a", 1), _row(1, "node:b", 1)]
    runs = mod.reconstruct_streak_runs(rows)
    assert all(r.left_censored is False for r in runs)


def test_length_distribution_excludes_left_censored_run() -> None:
    rows = [_row(0, "node:a", 3), _row(1, "node:b", 1), _row(2, "node:b", 2)]
    runs = mod.reconstruct_streak_runs(rows, first_run_left_censored=True)
    # a is left-censored (excluded), b is ongoing (excluded) -- nothing left to measure.
    assert mod.length_distribution(runs) == {}
    assert mod.qualification_rate_at(runs, 1) is None


def test_target_id_distribution_counts_completed_runs_only() -> None:
    rows = [
        _row(0, "node:a", 1),
        _row(1, "node:a", 2),
        _row(2, "node:a", 1),  # new run for node:a (count resets to 1 -- reconstruction
        # treats consecutive same-target rows as one run regardless of count resetting,
        # since update_dominance_streak never actually emits a reset-to-1 row for the
        # SAME target_id in real data -- this row exercises the pure grouping logic only.
        _row(3, "node:b", 1),  # ongoing
    ]
    runs = mod.reconstruct_streak_runs(rows)
    dist = mod.target_id_distribution(runs)
    assert dist == {"node:a": 1}  # b excluded (ongoing)
