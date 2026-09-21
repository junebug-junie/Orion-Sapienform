"""Deterministic unit tests for measure_outreach_reason_clusters.py (no DB)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_outreach_reason_clusters.py"
_spec = importlib.util.spec_from_file_location("measure_outreach_reason_clusters", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_outreach_reason_clusters"] = mod
_spec.loader.exec_module(mod)


def test_target_monoculture_fails_when_one_node_dominates() -> None:
    sends = [{"target_id": "node:athena"} for _ in range(8)] + [
        {"target_id": "node:circe"} for _ in range(2)
    ]
    result = mod.check_target_monoculture(sends)
    assert result["pass"] is False
    assert "node:athena" in result["detail"]


def test_target_monoculture_passes_when_spread() -> None:
    sends = (
        [{"target_id": "node:athena"} for _ in range(4)]
        + [{"target_id": "node:circe"} for _ in range(3)]
        + [{"target_id": "node:atlas"} for _ in range(3)]
    )
    assert mod.check_target_monoculture(sends)["pass"] is True


def test_cap_pin_fails_when_every_day_hits_cap() -> None:
    daily = {date(2026, 9, d): 4 for d in range(1, 11)}
    result = mod.check_cap_pin(daily, daily_cap=4)
    assert result["pass"] is False


def test_cap_pin_passes_when_any_day_below_cap() -> None:
    daily = {date(2026, 9, d): 4 for d in range(1, 10)}
    daily[date(2026, 9, 10)] = 3
    assert mod.check_cap_pin(daily, daily_cap=4)["pass"] is True


def test_content_gate_fails_at_zero() -> None:
    assert mod.check_content_gate(0)["pass"] is False
    assert mod.check_content_gate(2)["pass"] is True


def test_identity_coverage_unverifiable_without_id_keys() -> None:
    sends = [
        {"grounding": {"priors_count": 2, "curiosity_summaries": 1}},
        {"grounding": {"priors_count": 1, "curiosity_summaries": 0}},
    ]
    result = mod.check_identity_coverage(sends)
    assert result["pass"] is None
    assert "UNVERIFIED" in result["detail"]


def test_identity_coverage_skips_pre_patch_in_mixed_window() -> None:
    """One post-patch row must not fail because older rows lack ID keys."""
    sends = [
        {"grounding": {"priors_count": 2, "curiosity_summaries": 1}},
        {
            "grounding": {
                "priors_count": 1,
                "prior_ids": ["p1"],
                "curiosity_summaries": 0,
                "curiosity_content_ids": [],
            }
        },
    ]
    result = mod.check_identity_coverage(sends)
    assert result["pass"] is True
    assert "skipped_pre_patch=1" in result["detail"]


def test_identity_coverage_passes_when_aligned() -> None:
    sends = [
        {
            "grounding": {
                "priors_count": 2,
                "prior_ids": ["a", "b"],
                "curiosity_summaries": 1,
                "curiosity_content_ids": ["s1"],
            }
        }
    ]
    assert mod.check_identity_coverage(sends)["pass"] is True


def test_identity_coverage_fails_on_length_mismatch() -> None:
    sends = [
        {
            "grounding": {
                "priors_count": 2,
                "prior_ids": ["a"],
                "curiosity_summaries": 0,
                "curiosity_content_ids": [],
            }
        }
    ]
    assert mod.check_identity_coverage(sends)["pass"] is False


def test_repeat_content_is_informational() -> None:
    sends = [
        {"grounding": {"prior_ids": ["p1"], "curiosity_content_ids": ["c1"]}},
        {"grounding": {"prior_ids": ["p1"], "curiosity_content_ids": ["c2"]}},
    ]
    result = mod.check_repeat_content(sends)
    assert result["pass"] is None
    assert result["prior_repeats"]["p1"] == 2


def test_run_checks_returns_all_five() -> None:
    results = mod.run_checks(
        sends_7d=[{"target_id": "node:a", "grounding": {}}],
        sends_14d=[{"target_id": "node:a", "grounding": {}}],
        daily_sends_14d={date(2026, 9, 1): 1},
        tension_without_content_14d=0,
        daily_cap=4,
    )
    assert [r["name"] for r in results] == [
        "target_monoculture",
        "cap_pin",
        "content_gate_falsifiable",
        "identity_coverage",
        "repeat_content",
    ]
