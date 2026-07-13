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


def test_crystallization_ui_surfaces_retirement_candidates_and_deprecate_action() -> None:
    """Retirement surfacing (docs/superpowers/specs/2026-07-13-recall-followups-loop-
    retirement-saturation-gate-spec.md section 2): the review queue must pull in
    retirement-candidate rows from the list endpoint, badge them, and expose the
    existing deprecate endpoint as a click-through action. No automated browser
    harness exists for this page (confirmed: no JS/template UI test runner in
    services/orion-hub/tests beyond this text-smoke pattern), so this mirrors the
    existing wiring-smoke convention above rather than fabricating render coverage.
    """
    ui = (HUB_ROOT / "static" / "js" / "memory-crystallization-ui.js").read_text(encoding="utf-8")
    assert "retirement_candidate" in ui
    assert "decayed_activation" in ui
    assert "stale" in ui.lower()
    assert "/api/memory/crystallizations?status=active" in ui
    assert '/api/memory/crystallizations/${row.crystallization_id}/deprecate' in ui
    assert 'data-act="deprecate"' in ui
