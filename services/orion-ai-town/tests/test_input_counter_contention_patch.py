"""Gate tests for the input-counter contention fix (reduces OCC retries on sendInput)."""

from __future__ import annotations

from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
_PATCH = _SERVICE / "patches" / "orion-input-counter-contention.patch"
_APPLY = _SERVICE / "scripts" / "apply_upstream_patches.sh"


def test_patch_registered_in_apply_script():
    text = _APPLY.read_text(encoding="utf-8")
    assert "orion-input-counter-contention.patch" in text


def test_patch_registered_last():
    text = _APPLY.read_text(encoding="utf-8")
    order = [
        line.strip().strip('",')
        for line in text.splitlines()
        if line.strip().startswith('"orion-')
    ]
    assert order[-1] == "orion-input-counter-contention.patch"


def test_patch_touches_exactly_the_expected_files():
    patch = _PATCH.read_text(encoding="utf-8")
    assert patch.count("diff --git") == 4
    for path in (
        "convex/aiTown/insertInput.ts",
        "convex/aiTown/schema.ts",
        "convex/engine/abstractGame.ts",
        "convex/engine/schema.ts",
    ):
        assert path in patch


def test_patch_adds_input_counters_table():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "inputCounters: defineTable(inputCounter).index('engineId', ['engineId'])" in patch


def test_patch_adds_world_engine_map_table():
    patch = _PATCH.read_text(encoding="utf-8")
    assert "worldEngineMap: defineTable({" in patch


def test_patch_removes_read_before_increment_from_engine_insert_input():
    patch = _PATCH.read_text(encoding="utf-8")
    # The old read-last-input-then-increment pattern must be removed from
    # engineInsertInput's hot path -- it's what made every sendInput call
    # conflict with every other concurrent sendInput and with saveWorld.
    assert "-  const prevInput = await ctx.db" in patch
    assert "-  const number = prevInput ? prevInput.number + 1 : 0;" in patch
    assert "+  const number = await nextInputNumber(ctx, engineId);" in patch


def test_patch_lazily_seeds_counters_and_mapping_no_separate_migration():
    patch = _PATCH.read_text(encoding="utf-8")
    # Both new tables must be populated lazily on first use per
    # engine/world -- there is no separate migration script, so engines and
    # worlds that predate this patch must self-heal on the first call.
    assert "await ctx.db.insert('inputCounters'" in patch
    assert "await ctx.db.insert('worldEngineMap'" in patch
