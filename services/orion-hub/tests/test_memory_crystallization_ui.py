from __future__ import annotations

from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]


def test_crystallization_observatory_ui_wired() -> None:
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    ui = (HUB_ROOT / "static" / "js" / "memory-crystallization-ui.js").read_text(encoding="utf-8")
    assert "memorySubviewCrystallizations" in template
    assert "memory-crystallization-ui.js" in template
    assert "/api/memory/crystallizations/proposals" in ui
    assert "projection/health" in ui
