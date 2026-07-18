"""Deterministic unit tests for measure_capability_channel_health.py.

No DB, no network. Everything exercised here is pure: extract_channel_value
parses a raw capability_vectors JSON blob with no I/O, and
summarize_channel / frac_zero_or_subnormal / classify_channel operate on
plain synthetic float lists. Same sibling-module-by-file-path loading
pattern as test_measure_capability_salience_coupling.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_capability_channel_health.py"
_spec = importlib.util.spec_from_file_location("measure_capability_channel_health", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_capability_channel_health"] = mod
_spec.loader.exec_module(mod)


# ===========================================================================
# extract_channel_value -- pure JSON parsing, must never raise.
# ===========================================================================


def test_extract_channel_value_found() -> None:
    vectors = {
        "capability:transport": {"reliability_pressure": 0.42, "pressure": 0.1},
        "capability:llm_inference": {"reliability_pressure": 0.91},
    }
    assert mod.extract_channel_value(vectors, "capability:transport", "reliability_pressure") == 0.42
    assert mod.extract_channel_value(vectors, "capability:transport", "pressure") == 0.1
    assert mod.extract_channel_value(vectors, "capability:llm_inference", "reliability_pressure") == 0.91


def test_extract_channel_value_absent_channel_defaults_to_zero() -> None:
    vectors = {"capability:transport": {"pressure": 0.9}}
    assert mod.extract_channel_value(vectors, "capability:transport", "reliability_pressure") == 0.0


def test_extract_channel_value_absent_target_defaults_to_zero() -> None:
    vectors = {"capability:transport": {"pressure": 0.9}}
    assert mod.extract_channel_value(vectors, "capability:not_present", "pressure") == 0.0


def test_extract_channel_value_handles_json_string() -> None:
    raw = '{"capability:transport": {"reliability_pressure": 0.55}}'
    assert mod.extract_channel_value(raw, "capability:transport", "reliability_pressure") == 0.55


def test_extract_channel_value_none_never_raises() -> None:
    assert mod.extract_channel_value(None, "capability:transport", "pressure") == 0.0


def test_extract_channel_value_malformed_json_string_never_raises() -> None:
    assert mod.extract_channel_value("{not valid json", "capability:transport", "pressure") == 0.0


def test_extract_channel_value_wrong_shape_never_raises() -> None:
    # Not a dict at all.
    assert mod.extract_channel_value(["unexpected", "shape"], "capability:transport", "pressure") == 0.0
    # Dict, but the target entry is not itself a dict.
    assert mod.extract_channel_value(
        {"capability:transport": "not-a-dict"}, "capability:transport", "pressure"
    ) == 0.0
    # Target entry present, channel value is garbage.
    assert mod.extract_channel_value(
        {"capability:transport": {"pressure": "not-a-number"}}, "capability:transport", "pressure"
    ) == 0.0


# ===========================================================================
# is_zero_or_subnormal -- the whole point of this script vs. a plain `== 0.0`
# check.
# ===========================================================================


def test_is_zero_or_subnormal_catches_exact_zero() -> None:
    assert mod.is_zero_or_subnormal(0.0) is True


def test_is_zero_or_subnormal_catches_decayed_subnormal_float() -> None:
    # Mirrors the real live value observed in substrate_field_state
    # (~6.85e-322) -- numerically dead but not bitwise-zero.
    assert mod.is_zero_or_subnormal(1e-300) is True
    assert mod.is_zero_or_subnormal(6.85e-322) is True


def test_is_zero_or_subnormal_does_not_flag_real_signal() -> None:
    assert mod.is_zero_or_subnormal(0.3) is False
    assert mod.is_zero_or_subnormal(-0.3) is False
    # Just above the cutoff.
    assert mod.is_zero_or_subnormal(1e-50) is False


# ===========================================================================
# summarize_channel
# ===========================================================================


def test_summarize_channel_empty() -> None:
    dist = mod.summarize_channel([])
    assert dist.count == 0
    assert dist.median is None
    assert dist.p90 is None
    assert dist.max is None


def test_summarize_channel_known_values() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist = mod.summarize_channel(values)
    assert dist.count == 10
    assert dist.median == 0.55
    assert dist.max == 1.0
    assert abs(dist.p90 - 0.91) < 1e-9


# ===========================================================================
# frac_zero_or_subnormal
# ===========================================================================


def test_frac_zero_or_subnormal_known_values() -> None:
    values = [0.0, 1e-300, 0.2, 0.5]
    assert mod.frac_zero_or_subnormal(values) == 0.5
    assert mod.frac_zero_or_subnormal([]) == 0.0
    assert mod.frac_zero_or_subnormal([0.1, 0.2]) == 0.0
    assert mod.frac_zero_or_subnormal([0.0, 1e-320]) == 1.0


# ===========================================================================
# classify_channel -- live/dead verdict logic on synthetic distributions.
# ===========================================================================


def test_classify_channel_clearly_live() -> None:
    # Real, varying signal: spread well above LIVE_SPREAD_THRESHOLD (0.05).
    values = [0.05, 0.1, 0.3, 0.5, 0.9]
    dist = mod.summarize_channel(values)
    assert mod.classify_channel(dist) == "live"


def test_classify_channel_clearly_dead_pinned_near_zero() -> None:
    # Pinned near zero the whole window -- classic dead/decayed pattern.
    values = [0.0, 1e-300, 0.0, 6.85e-322, 0.0]
    dist = mod.summarize_channel(values)
    assert mod.classify_channel(dist) == "dead"


def test_classify_channel_empty_is_dead() -> None:
    dist = mod.summarize_channel([])
    assert mod.classify_channel(dist) == "dead"


def test_classify_channel_boundary_spread_is_dead_not_live() -> None:
    # max - median exactly at threshold (not strictly greater) -> dead.
    # median of [0.0, 0.0, 0.05] is 0.0, max is 0.05, spread == 0.05.
    values = [0.0, 0.0, 0.05]
    dist = mod.summarize_channel(values)
    assert dist.max - dist.median == 0.05
    assert mod.classify_channel(dist) == "dead"
