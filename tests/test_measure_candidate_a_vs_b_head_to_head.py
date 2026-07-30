"""Deterministic unit tests for measure_candidate_a_vs_b_head_to_head.py.

No DB, no network. Same sibling-module-by-file-path loading pattern as
test_measure_precision_weighted_salience_probe.py / test_measure_society_of_
mind_magnitude_probe.py -- lives in top-level `tests/`, not `scripts/
analysis/tests/`, since `scripts` is in pyproject.toml's `norecursedirs`
(confirmed defect for every sibling test under that path, documented in
those two files' own docstrings, not re-litigated here).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "measure_candidate_a_vs_b_head_to_head.py"
)
_spec = importlib.util.spec_from_file_location(
    "measure_candidate_a_vs_b_head_to_head", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_candidate_a_vs_b_head_to_head"] = mod
_spec.loader.exec_module(mod)


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
# replay_both_candidates -- exercises the REAL precision_weighted_salience()
# and magnitude_scorer() imports, not reimplementations.
# ===========================================================================


def test_replay_both_candidates_length_matches_qualifying_ticks() -> None:
    values = [0.1] * 25
    replay = mod.replay_both_candidates(values, "node:substrate.execution", min_samples=20, max_window=200)
    # ticks 0..18 (indices) are below min_samples-1=19 -- first qualifying
    # index is 19 (the 20th sample), so len == 25 - 19 == 6
    assert len(replay) == 6
    assert replay[0]["index"] == 19


def test_replay_both_candidates_b_magnitude_is_raw_clamped_value() -> None:
    values = [0.1] * 19 + [0.7]
    replay = mod.replay_both_candidates(values, "node:substrate.execution", min_samples=20, max_window=200)
    assert len(replay) == 1
    assert replay[0]["b_magnitude"] == 0.7
    assert replay[0]["value"] == 0.7


def test_replay_both_candidates_a_uses_expanding_window_not_full_history() -> None:
    # A real spike late in the series must not affect an earlier tick's own
    # precision estimate -- expanding window means tick i only ever sees
    # values[0:i+1], never future values.
    values = [0.05] * 20 + [0.9]  # spike is the LAST value only
    replay = mod.replay_both_candidates(values, "node:substrate.execution", min_samples=20, max_window=200)
    first_tick = replay[0]  # index 19, window = values[0:20], no spike yet
    assert first_tick["a_variance"] < 1e-6  # flat window, no spike included
    assert first_tick["value"] == 0.05


def test_replay_both_candidates_empty_input_returns_empty() -> None:
    assert mod.replay_both_candidates([], "node:substrate.execution", min_samples=20, max_window=200) == []


def test_replay_both_candidates_below_min_samples_returns_empty() -> None:
    values = [0.1] * 5
    replay = mod.replay_both_candidates(values, "node:substrate.execution", min_samples=20, max_window=200)
    assert replay == []


def test_replay_both_candidates_window_is_capped_to_match_production() -> None:
    # Code-review regression test (2026-07-30). Production
    # (AttentionRuntimeStore.load_prediction_error_history) only ever sees
    # the most recent `limit` real rows -- an unbounded expanding window
    # here would compute a different (larger, stale-inclusive) variance/
    # precision than the live selector actually produces once a reducer
    # exceeds max_window real rows within its retention window.
    #
    # Decisive fixture: a spike as the FIRST value, then 30 calm samples.
    # A capped window (max_window=10) should age the spike out of range by
    # the last tick; an uncapped window (max_window=1000) should still see it.
    values = [0.9] + [0.05] * 30
    capped = mod.replay_both_candidates(values, "node:substrate.execution", min_samples=20, max_window=10)
    uncapped = mod.replay_both_candidates(values, "node:substrate.execution", min_samples=20, max_window=1000)
    assert capped[-1]["a_variance"] < 1e-9  # spike aged out -- capped window is all-calm
    assert uncapped[-1]["a_variance"] > 1e-9  # uncapped window still includes the spike


# ===========================================================================
# top1_by_key
# ===========================================================================


def test_top1_by_key_picks_highest_scoring_entry() -> None:
    replay = [
        {"index": 0, "a_salience": 5.0},
        {"index": 1, "a_salience": 12.0},
        {"index": 2, "a_salience": 3.0},
    ]
    top = mod.top1_by_key(replay, "a_salience")
    assert top["index"] == 1


def test_top1_by_key_empty_input_returns_none() -> None:
    assert mod.top1_by_key([], "a_salience") is None


def test_top1_by_key_tie_returns_first_occurrence() -> None:
    # Documents the real, disclosed behavior the report's own tie-detection
    # branch relies on: Python's max() resolves ties to the first-encountered
    # item, not a real preference -- the report must not describe this as a
    # meaningful pick when scores are truly tied (see the report's own
    # "DISAGREE (tie-break case)" branch).
    replay = [
        {"index": 0, "b_magnitude": 1.0},
        {"index": 1, "b_magnitude": 1.0},
    ]
    top = mod.top1_by_key(replay, "b_magnitude")
    assert top["index"] == 0
