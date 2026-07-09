from __future__ import annotations

from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]


def test_crystallization_observatory_ui_wired() -> None:
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    ui = (HUB_ROOT / "static" / "js" / "memory-crystallization-ui.js").read_text(encoding="utf-8")
    memory_js = (HUB_ROOT / "static" / "js" / "memory.js").read_text(encoding="utf-8")
    assert "memorySubviewCrystallizations" in template
    assert "memory-crystallization-ui.js" in template
    assert "/api/memory/crystallizations/proposals" in ui
    assert "projection/health" in ui
    assert "OrionMemoryCrystallizationUI" in ui
    assert 'activateSubview("crystallizations")' in memory_js
    assert "memoryCrystallizationPanel" in memory_js


def test_crystallization_ui_shows_graphiti_projection_and_sync() -> None:
    ui = (HUB_ROOT / "static" / "js" / "memory-crystallization-ui.js").read_text(encoding="utf-8")
    assert "graphiti_episode_ids" in ui
    assert "/api/memory/graphiti/sync/" in ui
    assert "renderEvidence" in ui
    assert "window_novelty_max" in ui
