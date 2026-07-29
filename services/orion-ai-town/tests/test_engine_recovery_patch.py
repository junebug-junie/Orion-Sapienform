"""Gate tests for the engine-recovery patch's index-range fix and sizing probes.

Regression coverage for a bug found live 2026-07-29: `debugEngineState` and
`recoverFrozenEngine` used `.withIndex(eq engineId only).filter(gt number)`,
which can't prune the index range and forces a full scan of the engine's
entire input history (300k+ rows) instead of just the pending tail. Both
timed out ("too many system operations") regardless of `.take()` limit until
fixed to chain the range into `withIndex` directly.
"""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-engine-recovery.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def test_engine_recovery_patch_registered_in_apply_script():
    text = _APPLY.read_text(encoding="utf-8")
    assert "orion-engine-recovery.patch" in text


def test_debug_and_recover_use_indexed_range_not_post_filter():
    patch = _PATCH.read_text(encoding="utf-8")
    # The fixed form chains .gt('number', processed) into the same withIndex
    # call so Convex can prune the scan range.
    assert patch.count("q.eq('engineId', engine._id).gt('number', processed)") == 2
    # The old, broken form (index eq-only + a post-hoc .filter for the range)
    # must not reappear.
    assert ".filter((q) => q.gt(q.field('number'), processed))" not in patch


def test_debug_and_recover_are_bounded_by_limit():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "args: { limit: v.optional(v.number()) }" in patch
    assert ".take(limit)" in patch
    # The old unbounded form collected the whole (filtered) pending set;
    # the fixed form never collects the inputs index at all.
    assert "byInputNumber', (q) => q.eq('engineId', engine._id))\n      .filter" not in patch


def test_table_growth_sample_probe_present():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "tableGrowthSample" in patch
    assert "by_creation_time" in patch


def test_world_doc_size_probe_present():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "worldDocSizeProbe" in patch
