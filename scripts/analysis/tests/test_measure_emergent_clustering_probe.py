"""Deterministic unit tests for measure_emergent_clustering_probe.py.

No DB, no network. Everything here is pure: JSON-target-list parsing,
series alignment, Pearson correlation, union-find clustering, similarity
metrics, and window selection all operate on plain synthetic data. Same
sibling-module-by-file-path loading pattern as
test_measure_capability_salience_coupling.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_emergent_clustering_probe.py"
_spec = importlib.util.spec_from_file_location("measure_emergent_clustering_probe", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_emergent_clustering_probe"] = mod
_spec.loader.exec_module(mod)

UTC = timezone.utc
BASE = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)


# ===========================================================================
# extract_target_salience_map -- pure JSON parsing, must never raise.
# ===========================================================================


def test_extract_target_salience_map_merges_dominant_and_capability() -> None:
    dominant = [
        {"target_id": "node:athena", "salience_score": 0.7},
        {"target_id": "capability:transport", "salience_score": 0.5},
    ]
    capability = [{"target_id": "capability:transport", "salience_score": 0.5}]
    result = mod.extract_target_salience_map(dominant, capability)
    assert result == {"node:athena": 0.7, "capability:transport": 0.5}


def test_extract_target_salience_map_capability_fills_gap_not_present_in_dominant() -> None:
    """Defensive path: a target present in capability_targets but somehow
    absent from dominant_targets should still be captured."""
    dominant = [{"target_id": "node:athena", "salience_score": 0.7}]
    capability = [{"target_id": "capability:transport", "salience_score": 0.3}]
    result = mod.extract_target_salience_map(dominant, capability)
    assert result == {"node:athena": 0.7, "capability:transport": 0.3}


def test_extract_target_salience_map_handles_json_string() -> None:
    dominant = '[{"target_id": "node:athena", "salience_score": 0.42}]'
    result = mod.extract_target_salience_map(dominant, None)
    assert result == {"node:athena": 0.42}


def test_extract_target_salience_map_none_never_raises() -> None:
    assert mod.extract_target_salience_map(None, None) == {}


def test_extract_target_salience_map_malformed_never_raises() -> None:
    assert mod.extract_target_salience_map("{not valid json", "also not valid") == {}
    assert mod.extract_target_salience_map({"unexpected": "shape"}, ["not", "a", "dict"]) == {}
    assert mod.extract_target_salience_map(
        [{"target_id": "node:athena", "salience_score": "garbage"}], None
    ) == {"node:athena": 0.0}
    assert mod.extract_target_salience_map([{"salience_score": 0.9}], None) == {}


# ===========================================================================
# build_target_universe / align_series
# ===========================================================================


def test_build_target_universe_sorted_dedup() -> None:
    raw_maps = [{"b": 0.1, "a": 0.2}, {"c": 0.3}]
    assert mod.build_target_universe(raw_maps) == ["a", "b", "c"]


def test_align_series_absence_is_zero() -> None:
    raw_maps = [{"a": 0.5}, {"b": 0.9}, {"a": 0.2, "b": 0.1}]
    series = mod.align_series(raw_maps, ["a", "b"])
    assert series["a"] == [0.5, 0.0, 0.2]
    assert series["b"] == [0.0, 0.9, 0.1]


# ===========================================================================
# pearson_correlation
# ===========================================================================


def test_pearson_correlation_perfect_positive() -> None:
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    y = [0.2, 0.4, 0.6, 0.8, 1.0]
    r = mod.pearson_correlation(x, y)
    assert r is not None
    assert abs(r - 1.0) < 1e-9


def test_pearson_correlation_perfect_negative() -> None:
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    y = [0.5, 0.4, 0.3, 0.2, 0.1]
    r = mod.pearson_correlation(x, y)
    assert r is not None
    assert abs(r - (-1.0)) < 1e-9


def test_pearson_correlation_degenerate_flat_series_returns_none() -> None:
    x = [0.5, 0.5, 0.5, 0.5]
    y = [0.1, 0.2, 0.3, 0.4]
    assert mod.pearson_correlation(x, y) is None


def test_pearson_correlation_too_short_returns_none() -> None:
    assert mod.pearson_correlation([0.1], [0.2]) is None
    assert mod.pearson_correlation([], []) is None


def test_pearson_correlation_mismatched_lengths_returns_none() -> None:
    assert mod.pearson_correlation([0.1, 0.2], [0.1]) is None


def test_pearson_correlation_clamped_to_valid_range() -> None:
    x = [0.1, 0.5, 0.2, 0.9, 0.3]
    y = [0.9, 0.1, 0.8, 0.05, 0.7]
    r = mod.pearson_correlation(x, y)
    assert r is not None
    assert -1.0 <= r <= 1.0


# ===========================================================================
# compute_correlation_matrix / cluster_by_correlation / edge_set_at_threshold
# ===========================================================================


def test_compute_correlation_matrix_deterministic_key_order() -> None:
    series = {
        "z": [0.1, 0.2, 0.3, 0.4],
        "a": [0.1, 0.2, 0.3, 0.4],
    }
    matrix = mod.compute_correlation_matrix(["z", "a"], series)
    assert ("a", "z") in matrix
    assert ("z", "a") not in matrix
    assert abs(matrix[("a", "z")] - 1.0) < 1e-9


def test_cluster_by_correlation_groups_correlated_targets() -> None:
    series = {
        "a": [0.1, 0.5, 0.2, 0.9],
        "b": [0.1, 0.5, 0.2, 0.9],  # perfectly correlated with a
        "c": [0.9, 0.1, 0.8, 0.0],  # uncorrelated-ish with a/b
    }
    target_ids = ["a", "b", "c"]
    matrix = mod.compute_correlation_matrix(target_ids, series)
    clusters = mod.cluster_by_correlation(target_ids, matrix, threshold=0.9)
    # a and b must land in the same cluster; c should not be forced in.
    cluster_containing_a = next(c for c in clusters if "a" in c)
    assert "b" in cluster_containing_a


def test_cluster_by_correlation_degenerate_target_is_singleton() -> None:
    series = {
        "a": [0.5, 0.5, 0.5, 0.5],  # flat -> degenerate, correlation always None
        "b": [0.1, 0.9, 0.2, 0.8],
    }
    target_ids = ["a", "b"]
    matrix = mod.compute_correlation_matrix(target_ids, series)
    clusters = mod.cluster_by_correlation(target_ids, matrix, threshold=0.5)
    assert ["a"] in clusters


def test_edge_set_at_threshold_excludes_none_and_below_threshold() -> None:
    matrix = {("a", "b"): 0.9, ("a", "c"): 0.2, ("b", "c"): None}
    edges = mod.edge_set_at_threshold(matrix, threshold=0.5)
    assert edges == {("a", "b")}


# ===========================================================================
# jaccard_similarity
# ===========================================================================


def test_jaccard_similarity_identical_sets() -> None:
    a = {("x", "y"), ("y", "z")}
    assert mod.jaccard_similarity(a, a) == 1.0


def test_jaccard_similarity_disjoint_sets() -> None:
    a = {("x", "y")}
    b = {("p", "q")}
    assert mod.jaccard_similarity(a, b) == 0.0


def test_jaccard_similarity_partial_overlap() -> None:
    a = {("x", "y"), ("y", "z")}
    b = {("x", "y"), ("p", "q")}
    # intersection = 1, union = 3
    assert abs(mod.jaccard_similarity(a, b) - (1 / 3)) < 1e-9


def test_jaccard_similarity_both_empty_is_none() -> None:
    assert mod.jaccard_similarity(set(), set()) is None


# ===========================================================================
# correlation_of_correlations
# ===========================================================================


def test_correlation_of_correlations_identical_matrices() -> None:
    matrix = {("a", "b"): 0.9, ("a", "c"): 0.1, ("b", "c"): -0.3, ("a", "d"): 0.5}
    value, n = mod.correlation_of_correlations(matrix, dict(matrix))
    assert n == 4
    assert value is not None
    assert abs(value - 1.0) < 1e-9


def test_correlation_of_correlations_too_few_common_pairs_is_none() -> None:
    matrix_a = {("a", "b"): 0.9, ("a", "c"): 0.1}
    matrix_b = {("a", "b"): 0.9}
    value, n = mod.correlation_of_correlations(matrix_a, matrix_b)
    assert value is None
    assert n == 1


def test_correlation_of_correlations_excludes_none_valued_pairs() -> None:
    matrix_a = {("a", "b"): 0.9, ("a", "c"): None, ("b", "c"): 0.1, ("a", "d"): 0.5}
    matrix_b = {("a", "b"): 0.8, ("a", "c"): 0.2, ("b", "c"): 0.05, ("a", "d"): 0.4}
    value, n = mod.correlation_of_correlations(matrix_a, matrix_b)
    # ("a", "c") is None in matrix_a -> excluded, leaving 3 usable pairs.
    assert n == 3
    assert value is not None


# ===========================================================================
# classify_similarity -- explicit documented bands
# ===========================================================================


def test_classify_similarity_insufficient_pairs() -> None:
    assert mod.classify_similarity(0.9, 0.9, 2) == "INCONCLUSIVE_INSUFFICIENT_PAIRS"
    assert mod.classify_similarity(None, 0.9, 5) == "INCONCLUSIVE_INSUFFICIENT_PAIRS"


def test_classify_similarity_identical_trivial() -> None:
    assert mod.classify_similarity(1.0, 1.0, 5) == "IDENTICAL_TRIVIAL"


def test_classify_similarity_random_low_corr_of_corr() -> None:
    assert mod.classify_similarity(0.1, 0.5, 5) == "RANDOM"


def test_classify_similarity_random_low_jaccard() -> None:
    assert mod.classify_similarity(0.8, 0.05, 5) == "RANDOM"


def test_classify_similarity_recognizable_similarity() -> None:
    assert mod.classify_similarity(0.7, 0.4, 5) == "RECOGNIZABLE_SIMILARITY"


def test_classify_similarity_ambiguous_falls_between_bands() -> None:
    # corr_of_corr strong but jaccard is a full 1.0 without also clearing the
    # exact IDENTICAL_TRIVIAL bar (corr_of_corr < 0.999) -> ambiguous, not
    # forced into either MET or NOT MET band.
    assert mod.classify_similarity(0.7, 1.0, 5) == "AMBIGUOUS"


# ===========================================================================
# top1_winner / top1_winner_distribution / membership_frequency
# ===========================================================================


def test_top1_winner_picks_highest_score() -> None:
    assert mod.top1_winner({"a": 0.5, "b": 0.9, "c": 0.3}) == "b"


def test_top1_winner_deterministic_tiebreak_alphabetical() -> None:
    assert mod.top1_winner({"z": 0.5, "a": 0.5}) == "a"


def test_top1_winner_empty_map_is_none() -> None:
    assert mod.top1_winner({}) is None


def test_top1_winner_distribution_counts() -> None:
    raw_maps = [{"a": 0.9}, {"a": 0.8}, {"b": 0.99}]
    dist = mod.top1_winner_distribution(raw_maps)
    assert dist["a"] == 2
    assert dist["b"] == 1


def test_membership_frequency_fraction_present() -> None:
    raw_maps = [{"a": 0.5}, {"a": 0.1, "b": 0.2}, {"b": 0.9}]
    freq = mod.membership_frequency(raw_maps, ["a", "b"])
    assert abs(freq["a"] - (2 / 3)) < 1e-9
    assert abs(freq["b"] - (2 / 3)) < 1e-9


def test_membership_frequency_empty_history() -> None:
    freq = mod.membership_frequency([], ["a"])
    assert freq == {"a": 0.0}


# ===========================================================================
# choose_windows
# ===========================================================================


def test_choose_windows_ample_span_anchors_start_and_end() -> None:
    min_ts = BASE
    max_ts = BASE + timedelta(hours=72)
    windows = mod.choose_windows(min_ts, max_ts, window_hours=24.0, gap_hours=12.0)
    assert windows is not None
    win_a, win_b = windows
    assert win_a.start == min_ts
    assert win_a.end == min_ts + timedelta(hours=24)
    assert win_b.end == max_ts
    assert win_b.start == max_ts - timedelta(hours=24)
    # non-overlapping, with the requested gap preserved
    assert win_a.end <= win_b.start


def test_choose_windows_shrinks_gracefully_for_tight_span() -> None:
    min_ts = BASE
    max_ts = BASE + timedelta(hours=10)  # too short for 24h+12h+24h
    windows = mod.choose_windows(min_ts, max_ts, window_hours=24.0, gap_hours=12.0, min_window_hours=4.0)
    assert windows is not None
    win_a, win_b = windows
    assert win_a.start == min_ts
    assert win_b.end == max_ts
    assert win_a.end <= win_b.start  # still non-overlapping
    assert (win_a.end - win_a.start).total_seconds() / 3600.0 >= 4.0
    assert (win_b.end - win_b.start).total_seconds() / 3600.0 >= 4.0


def test_choose_windows_insufficient_span_returns_none() -> None:
    min_ts = BASE
    max_ts = BASE + timedelta(hours=2)  # less than 2 * min_window_hours=4.0
    windows = mod.choose_windows(min_ts, max_ts, window_hours=24.0, gap_hours=12.0, min_window_hours=4.0)
    assert windows is None


def test_choose_windows_none_timestamps_returns_none() -> None:
    assert mod.choose_windows(None, None) is None


def test_choose_windows_inverted_range_returns_none() -> None:
    assert mod.choose_windows(BASE, BASE - timedelta(hours=1)) is None
