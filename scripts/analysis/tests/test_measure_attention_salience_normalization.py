"""Deterministic unit tests for measure_attention_salience_normalization.py.

No DB, no network. Everything here is pure: JSON-target-list parsing, series
alignment, per-channel mean/stddev, z-score series, normalized-ranking
argmax, and the classification bands all operate on plain synthetic data.
Same sibling-module-by-file-path loading pattern as
test_measure_emergent_clustering_probe.py.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_attention_salience_normalization.py"
_spec = importlib.util.spec_from_file_location("measure_attention_salience_normalization", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_attention_salience_normalization"] = mod
_spec.loader.exec_module(mod)


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


def test_extract_target_salience_map_handles_json_string() -> None:
    dominant = '[{"target_id": "node:athena", "salience_score": 0.42}]'
    result = mod.extract_target_salience_map(dominant, None)
    assert result == {"node:athena": 0.42}


def test_extract_target_salience_map_none_never_raises() -> None:
    assert mod.extract_target_salience_map(None, None) == {}


def test_extract_target_salience_map_malformed_never_raises() -> None:
    assert mod.extract_target_salience_map("{not valid json", "also not valid") == {}
    assert mod.extract_target_salience_map(
        [{"target_id": "node:athena", "salience_score": "garbage"}], None
    ) == {"node:athena": 0.0}


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
# channel_mean_std -- degenerate-variance detection.
# ===========================================================================


def test_channel_mean_std_normal_series() -> None:
    mean, std, degenerate = mod.channel_mean_std([0.1, 0.2, 0.3, 0.4, 0.5])
    assert abs(mean - 0.3) < 1e-9
    assert std > 0.0
    assert degenerate is False


def test_channel_mean_std_flat_series_is_degenerate() -> None:
    mean, std, degenerate = mod.channel_mean_std([1.0, 1.0, 1.0, 1.0])
    assert abs(mean - 1.0) < 1e-9
    assert std == 0.0
    assert degenerate is True


def test_channel_mean_std_empty_series_is_degenerate() -> None:
    mean, std, degenerate = mod.channel_mean_std([])
    assert degenerate is True


def test_channel_mean_std_near_constant_saturated_channel_is_degenerate() -> None:
    """The real repo finding this script exists to check: a channel that
    saturates to ~1.0 and stays there (field:recent_perturbations-shaped)
    must be flagged degenerate, not treated as having real variance."""
    values = [1.0] * 999 + [0.9999999999999]
    mean, std, degenerate = mod.channel_mean_std(values)
    assert degenerate is True


# ===========================================================================
# compute_zscore_series
# ===========================================================================


def test_compute_zscore_series_normal_channel() -> None:
    target_ids = ["a"]
    raw_series = {"a": [0.0, 1.0, 2.0, 3.0, 4.0]}
    z_series, stats = mod.compute_zscore_series(target_ids, raw_series)
    mean, std, degenerate = stats["a"]
    assert degenerate is False
    assert abs(mean - 2.0) < 1e-9
    # z-scores should be symmetric around zero for a linear ramp.
    assert abs(sum(z_series["a"])) < 1e-9
    assert all(v is not None for v in z_series["a"])


def test_compute_zscore_series_degenerate_channel_is_all_none() -> None:
    target_ids = ["a"]
    raw_series = {"a": [1.0, 1.0, 1.0, 1.0]}
    z_series, stats = mod.compute_zscore_series(target_ids, raw_series)
    assert stats["a"][2] is True  # degenerate
    assert z_series["a"] == [None, None, None, None]


def test_compute_zscore_series_multi_channel_independent() -> None:
    target_ids = ["a", "b"]
    raw_series = {
        "a": [0.0, 1.0, 2.0],
        "b": [1.0, 1.0, 1.0],  # degenerate
    }
    z_series, stats = mod.compute_zscore_series(target_ids, raw_series)
    assert stats["a"][2] is False
    assert stats["b"][2] is True
    assert all(v is not None for v in z_series["a"])
    assert z_series["b"] == [None, None, None]


# ===========================================================================
# top1_winner_raw / top1_winner_distribution_raw
# ===========================================================================


def test_top1_winner_raw_picks_highest_score() -> None:
    assert mod.top1_winner_raw({"a": 0.5, "b": 0.9, "c": 0.3}) == "b"


def test_top1_winner_raw_deterministic_tiebreak_alphabetical() -> None:
    assert mod.top1_winner_raw({"z": 0.5, "a": 0.5}) == "a"


def test_top1_winner_raw_empty_map_is_none() -> None:
    assert mod.top1_winner_raw({}) is None


def test_top1_winner_distribution_raw_counts() -> None:
    raw_maps = [{"a": 0.9}, {"a": 0.8}, {"b": 0.99}]
    dist = mod.top1_winner_distribution_raw(raw_maps)
    assert dist["a"] == 2
    assert dist["b"] == 1


# ===========================================================================
# top1_winner_distribution_normalized -- the core new logic.
# ===========================================================================


def test_top1_winner_distribution_normalized_picks_highest_z() -> None:
    target_ids = ["a", "b"]
    # a: mean 1, std 1 -> z = [-1, 0, 1]
    # b: mean 0, std 1 -> z = [1, 0, -1]  (b wins tick 0, tie tick 1, a wins tick 2)
    z_series = {
        "a": [-1.0, 0.0, 1.0],
        "b": [1.0, 0.0, -1.0],
    }
    dist, n_no_winner = mod.top1_winner_distribution_normalized(target_ids, z_series, 3)
    assert n_no_winner == 0
    assert dist["b"] == 1  # tick 0
    assert dist["a"] == 2  # tick 1 (tiebreak alphabetical -> a), tick 2


def test_top1_winner_distribution_normalized_excludes_degenerate_channel() -> None:
    """A degenerate channel (all-None z-series) must never win -- even if its
    raw value would have been highest every tick under the old ranking."""
    target_ids = ["saturated", "real"]
    z_series = {
        "saturated": [None, None, None],
        "real": [-0.5, 0.2, 1.0],
    }
    dist, n_no_winner = mod.top1_winner_distribution_normalized(target_ids, z_series, 3)
    assert n_no_winner == 0
    assert "saturated" not in dist
    assert dist["real"] == 3


def test_top1_winner_distribution_normalized_all_degenerate_tick_has_no_winner() -> None:
    target_ids = ["a", "b"]
    z_series = {
        "a": [None, 0.5],
        "b": [None, -0.5],
    }
    dist, n_no_winner = mod.top1_winner_distribution_normalized(target_ids, z_series, 2)
    assert n_no_winner == 1
    assert dist["a"] == 1


# ===========================================================================
# classify_normalization_effect -- explicit documented bands.
# ===========================================================================


def test_classify_normalization_effect_insufficient_data_too_few_rows() -> None:
    raw = Counter({"a": 100})
    norm = Counter({"a": 100})
    result = mod.classify_normalization_effect(raw, 100, norm, 100)
    assert result == "INSUFFICIENT_DATA"


def test_classify_normalization_effect_insufficient_data_zero_valid_normalized_ticks() -> None:
    raw = Counter({"a": 1000})
    norm = Counter()
    result = mod.classify_normalization_effect(raw, 1000, norm, 0)
    assert result == "INSUFFICIENT_DATA"


def test_classify_normalization_effect_not_met_same_winner_still_dominant() -> None:
    raw = Counter({"a": 900, "b": 100})
    norm = Counter({"a": 800, "b": 200})
    result = mod.classify_normalization_effect(raw, 1000, norm, 1000)
    assert result == "NOT_MET"


def test_classify_normalization_effect_monoculture_shifted() -> None:
    """This is the real live finding this script produced against
    substrate_attention_frames on 2026-07-28: raw dominant channel is
    degenerate, gets excluded, but a DIFFERENT single channel takes over at
    >= 50% share under normalization -- not a fix, a relocation."""
    raw = Counter({"field:recent_perturbations": 1000})
    norm = Counter({"node:atlas": 552, "node:circe": 200, "capability:llm_inference": 150, "node:athena": 98})
    result = mod.classify_normalization_effect(raw, 1000, norm, 1000)
    assert result == "NOT_MET_MONOCULTURE_SHIFTED"


def test_classify_normalization_effect_met_real_diversification() -> None:
    raw = Counter({"a": 900, "b": 100})
    norm = Counter({"a": 300, "b": 250, "c": 250, "d": 200})
    result = mod.classify_normalization_effect(raw, 1000, norm, 1000)
    assert result == "MET"


def test_classify_normalization_effect_ambiguous_partial_drop_no_real_diversity() -> None:
    # Winner's share drops from 90% to 45% (a real 0.45 drop, clears the 0.30
    # threshold) AND stays below the 0.5 monoculture threshold, but the
    # remaining 55% is spread thin enough that only ONE channel (the winner
    # itself) clears the 10% meaningful-diversity bar -- not "at least two
    # distinct channels", so this must land in AMBIGUOUS, not MET.
    raw = Counter({"a": 270, "b": 30})
    norm = Counter({"a": 135, "b": 27, "c": 27, "d": 27, "e": 27, "f": 27, "g": 27, "h": 3})
    result = mod.classify_normalization_effect(raw, 300, norm, 300)
    assert result == "AMBIGUOUS"


# ---------------------------------------------------------------------------
# Exact-boundary conditions -- the code uses `>=` (inclusive) everywhere for
# MONOCULTURE_SHARE_THRESHOLD, DIVERSIFICATION_DROP_THRESHOLD, and
# MEANINGFUL_DIVERSITY_SHARE. These tests pin that inclusive behavior down so
# a future accidental `>` edit gets caught instead of silently flipping the
# verdict at the exact boundary.
# ---------------------------------------------------------------------------


def test_classify_normalization_effect_norm_share_exactly_at_monoculture_threshold() -> None:
    # norm_share == 0.5 exactly (the MONOCULTURE_SHARE_THRESHOLD) must still
    # count as a monoculture (inclusive >=), not fall through to MET/AMBIGUOUS.
    raw = Counter({"a": 900, "b": 100})
    norm = Counter({"a": 500, "b": 500})
    result = mod.classify_normalization_effect(raw, 1000, norm, 1000)
    assert result == "NOT_MET"


def test_classify_normalization_effect_drop_clears_diversification_threshold() -> None:
    # (raw_share - norm_share) clears DIVERSIFICATION_DROP_THRESHOLD (0.30)
    # with norm_share below the monoculture threshold and >= 2 meaningful
    # channels -> must count as MET (the code's comparison is inclusive >=).
    # Counts are chosen as exact dyadic fractions (denominator 1600, a power
    # of two) specifically so raw_share/norm_share/drop are bit-exact in IEEE
    # float -- 700/1000-style decimal fractions are NOT exactly representable
    # in binary float (e.g. `0.7 - 0.4 == 0.29999999999999993`, not `0.3`),
    # so pinning this test to a literal boundary of "== 0.30" would be
    # testing float-rounding noise, not the `>=` comparison this test exists
    # to cover. Verified: raw_share=0.75, norm_share=0.4375, drop=0.3125.
    raw = Counter({"a": 1200, "b": 400})  # raw_share = 0.75
    norm = Counter({"a": 700, "b": 500, "c": 400})  # norm_share = 0.4375, drop = 0.3125
    result = mod.classify_normalization_effect(raw, 1600, norm, 1600)
    assert result == "MET"


def test_classify_normalization_effect_channel_share_exactly_at_meaningful_diversity_bar() -> None:
    # A second channel at exactly MEANINGFUL_DIVERSITY_SHARE (0.10) must count
    # toward the "at least two distinct channels" MET requirement (inclusive
    # >=), not be excluded for landing exactly on the boundary.
    raw = Counter({"a": 900, "b": 100})  # raw_share = 0.90
    norm = Counter({"a": 450, "b": 100, "c": 90, "d": 90, "e": 90, "f": 90, "g": 90})
    # norm_share = 0.45 (< 0.5), drop = 0.45 (>= 0.30); meaningful channels:
    # a=45% and b=10% exactly -> 2 channels clear the >=10% bar, c-g at 9% each do not.
    result = mod.classify_normalization_effect(raw, 1000, norm, 1000)
    assert result == "MET"
