"""Deterministic tests for replay_curiosity_realized_nats.py. No DB, no graph."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_MODULE_PATH = Path(__file__).resolve().parents[1] / "replay_curiosity_realized_nats.py"
_spec = importlib.util.spec_from_file_location("replay_curiosity_realized_nats", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["replay_curiosity_realized_nats"] = mod
_spec.loader.exec_module(mod)

from orion.curiosity.value import kl_nats  # noqa: E402


def test_distribution_keeps_unknown_apart_from_zero() -> None:
    d = mod.distribution([None, 0.0, 0.0, 0.2, 0.4])
    assert (d.n, d.n_null, d.n_zero) == (5, 1, 2)
    assert d.fraction_zero == 0.5
    assert d.mean == pytest.approx(0.15)
    assert d.p90 == 0.4


def test_all_zero_is_degenerate_and_no_data_is_not_a_pass() -> None:
    assert mod.verdict(mod.distribution([0.0] * 10), max_days=60)[0].startswith("DEGENERATE")
    assert mod.verdict(mod.distribution([None, None]), max_days=60)[0] == "NO DATA"


def test_sparse_but_real_signal_gets_a_sample_size() -> None:
    values = [0.0] * 8 + [0.3, 0.5]  # most runs move nothing, some move a lot
    text, need = mod.verdict(mod.distribution(values), max_days=60)
    assert need is not None and need > 0
    assert text.startswith("USABLE") or text.startswith("DEGENERATE at this cadence")


def test_runs_needed_grows_with_noise() -> None:
    quiet = mod.runs_per_arm_to_detect_doubling(0.1, 0.05)
    noisy = mod.runs_per_arm_to_detect_doubling(0.1, 0.3)
    assert quiet < noisy
    assert mod.runs_per_arm_to_detect_doubling(0.0, 0.1) is None


def test_graph_summary_counts_unmoved_tests_it_cannot_see() -> None:
    priors = [
        {"prior_id": "a", "confidence": "0.92", "times_tested": "3"},
        {"prior_id": "b", "confidence": "0.55", "times_tested": 2},
    ]
    revisions = [{"prior_id": "a", "from_confidence": "0.95", "to_confidence": "0.92"}]
    g = mod.summarize_graph(priors, revisions)
    assert (g.total_tests, g.revisions, g.unmoved_tests_estimate) == (5, 1, 4)
    assert g.revision_nats.mean == pytest.approx(kl_nats(0.92, 0.95))
    # 0.92 (current) and 0.92 (revision target) are off the 0.05 grid; 0.55, 0.95 on it.
    assert g.on_005_grid == pytest.approx(2 / 4)


def test_arm_comparison_is_value_minus_uncertainty_with_a_ci() -> None:
    rows = (
        [{"arm": "value_order", "realized_nats": v} for v in (0.2, 0.3, 0.25, 0.35)]
        + [{"arm": "uncertainty_order", "realized_nats": v} for v in (0.1, 0.0, 0.15, 0.05)]
        + [{"arm": None, "realized_nats": 9.9}]
    )
    cmp = mod.compare_arms(rows)
    assert cmp.difference == pytest.approx(0.275 - 0.075)
    lo, hi = cmp.ci95
    assert lo < cmp.difference < hi
    assert cmp.value.n == 4 and cmp.uncertainty.n == 4


def test_failed_turns_are_counted_per_arm_not_read_as_zero() -> None:
    rows = (
        [{"arm": "value_order", "realized_nats": v, "turn_ok": True} for v in (0.2, 0.3)]
        + [{"arm": "value_order", "realized_nats": 0.0, "turn_ok": False}] * 3
        + [{"arm": "uncertainty_order", "realized_nats": v, "turn_ok": True} for v in (0.1, 0.05)]
    )
    cmp = mod.compare_arms(rows)
    assert cmp.value.n == 2 and cmp.value.n_zero == 0
    assert cmp.value.mean == pytest.approx(0.25)
    assert mod.failed_turns_by_arm(rows) == {"value_order": 3, "uncertainty_order": 0}
    # Only an explicit failure is left out; a row with no flag is kept.
    assert mod.completed_turns([{"realized_nats": 0.1}]) == [{"realized_nats": 0.1}]


def test_unknown_runs_are_counted_by_the_reason_hub_recorded() -> None:
    rows = [
        {"arm": "value_order", "realized_nats": None, "turn_ok": True, "unknown_reason": "no_start_snapshot"},
        {"arm": "value_order", "realized_nats": None, "turn_ok": True, "unknown_reason": "no_start_snapshot"},
        {"arm": "uncertainty_order", "realized_nats": None, "turn_ok": True, "unknown_reason": "start_stamped_by_this_run"},
        {"arm": "uncertainty_order", "realized_nats": 0.0, "turn_ok": True, "unknown_reason": None},
        {"arm": "uncertainty_order", "realized_nats": None, "turn_ok": False, "unknown_reason": "no_end_snapshot"},
    ]
    # The failed turn is left out here too; a row with no reason says so.
    assert mod.unknown_by_reason(rows) == {"no_start_snapshot": 2, "start_stamped_by_this_run": 1}
    assert mod.unknown_by_reason([{"realized_nats": None}]) == {"unrecorded": 1}


def test_graph_summary_reads_float_string_counts_and_flags_the_row_cap(monkeypatch) -> None:
    # Review finding: int("2.0") raised and silently dropped that prior's tests.
    priors = [{"prior_id": "a", "times_tested": "2.0"}, {"prior_id": "b", "times_tested": 3}]
    g = mod.summarize_graph(priors, [])
    assert g.total_tests == 5 and not g.truncated
    monkeypatch.setattr(mod, "ATLAS_PRIORS_LIMIT", 2)
    assert mod.summarize_graph(priors, []).truncated
