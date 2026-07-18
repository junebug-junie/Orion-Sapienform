"""Deterministic unit tests for measure_capability_salience_coupling.py.

No DB, no network. Everything exercised here is pure: extract_salience_for_target
parses a raw capability_targets JSON blob with no I/O, and summarize_salience /
frac_ge / threshold_sweep operate on plain synthetic float lists. Same
sibling-module-by-file-path loading pattern as
test_measure_origination_gate.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_capability_salience_coupling.py"
_spec = importlib.util.spec_from_file_location("measure_capability_salience_coupling", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_capability_salience_coupling"] = mod
_spec.loader.exec_module(mod)


# ===========================================================================
# extract_salience_for_target -- pure JSON parsing, must never raise.
# ===========================================================================


def test_extract_salience_for_target_found() -> None:
    targets = [
        {"target_id": "capability:transport", "salience_score": 0.42},
        {"target_id": "capability:llm_inference", "salience_score": 0.91},
    ]
    assert mod.extract_salience_for_target(targets, "capability:transport") == 0.42
    assert mod.extract_salience_for_target(targets, "capability:llm_inference") == 0.91


def test_extract_salience_for_target_absent_defaults_to_zero() -> None:
    """Absent from the tick's list means below min_salience -- 0.0, not a skip."""
    targets = [{"target_id": "capability:transport", "salience_score": 0.9}]
    assert mod.extract_salience_for_target(targets, "capability:not_present") == 0.0


def test_extract_salience_for_target_handles_json_string() -> None:
    raw = '[{"target_id": "capability:transport", "salience_score": 0.55}]'
    assert mod.extract_salience_for_target(raw, "capability:transport") == 0.55


def test_extract_salience_for_target_none_never_raises() -> None:
    assert mod.extract_salience_for_target(None, "capability:transport") == 0.0


def test_extract_salience_for_target_malformed_json_string_never_raises() -> None:
    assert mod.extract_salience_for_target("{not valid json", "capability:transport") == 0.0


def test_extract_salience_for_target_wrong_shape_never_raises() -> None:
    # Not a list at all.
    assert mod.extract_salience_for_target({"unexpected": "shape"}, "capability:transport") == 0.0
    # A list, but entries are not dicts.
    assert mod.extract_salience_for_target(["not", "a", "dict"], "capability:transport") == 0.0
    # Entry matches target_id but salience_score is garbage.
    assert mod.extract_salience_for_target(
        [{"target_id": "capability:transport", "salience_score": "not-a-number"}],
        "capability:transport",
    ) == 0.0
    # Entry is missing target_id entirely.
    assert mod.extract_salience_for_target([{"salience_score": 0.9}], "capability:transport") == 0.0


# ===========================================================================
# summarize_salience
# ===========================================================================


def test_summarize_salience_empty() -> None:
    dist = mod.summarize_salience([])
    assert dist.count == 0
    assert dist.median is None
    assert dist.p90 is None
    assert dist.max is None


def test_summarize_salience_known_values() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist = mod.summarize_salience(values)
    assert dist.count == 10
    # median of an even-length list is the mean of the two middle values.
    assert dist.median == 0.55
    assert dist.max == 1.0
    # p90 via the script's own linear-interpolation percentile: pos = 0.9 * 9 = 8.1
    # -> ordered[8] + 0.1 * (ordered[9] - ordered[8]) = 0.9 + 0.1 * 0.1 = 0.91
    assert abs(dist.p90 - 0.91) < 1e-9


# ===========================================================================
# frac_ge
# ===========================================================================


def test_frac_ge_known_values() -> None:
    values = [0.05, 0.2, 0.5, 0.8]
    assert mod.frac_ge(values, 0.10) == 0.75  # 0.2, 0.5, 0.8 qualify
    assert mod.frac_ge(values, 0.70) == 0.25  # only 0.8 qualifies
    assert mod.frac_ge(values, 0.0) == 1.0
    assert mod.frac_ge([], 0.5) == 0.0


# ===========================================================================
# threshold_sweep -- monotonically non-increasing as threshold rises. This is
# a real invariant of frac_ge, not just eyeballed: check it holds on two
# different synthetic distributions.
# ===========================================================================


def _assert_monotonic_non_increasing(sweep: dict) -> None:
    ordered_thresholds = sorted(sweep.keys())
    fracs = [sweep[t] for t in ordered_thresholds]
    for earlier, later in zip(fracs, fracs[1:]):
        assert earlier >= later, f"threshold_sweep must be non-increasing, got {sweep}"


def test_threshold_sweep_monotonic_uniform_spread() -> None:
    values = [round(i * 0.05, 2) for i in range(21)]  # 0.0, 0.05, ..., 1.0
    sweep = mod.threshold_sweep(values)
    assert set(sweep.keys()) == {0.10, 0.25, 0.45, 0.70}
    _assert_monotonic_non_increasing(sweep)
    assert mod.frac_ge(values, 0.10) >= mod.frac_ge(values, 0.70)


def test_threshold_sweep_monotonic_clustered_low() -> None:
    values = [0.01, 0.02, 0.03, 0.04, 0.9]
    sweep = mod.threshold_sweep(values)
    _assert_monotonic_non_increasing(sweep)
    assert mod.frac_ge(values, 0.10) >= mod.frac_ge(values, 0.70)
    # Only the 0.9 outlier clears every threshold in DEFAULT_THRESHOLDS.
    assert sweep[0.70] == 0.2


def test_threshold_sweep_empty_values() -> None:
    sweep = mod.threshold_sweep([])
    assert all(v == 0.0 for v in sweep.values())
