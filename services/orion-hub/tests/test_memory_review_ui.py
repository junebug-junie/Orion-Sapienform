"""Hub memory review panel exposes card metadata editors and cards-aware recall profiles."""

from __future__ import annotations

from pathlib import Path


HUB_ROOT = Path(__file__).resolve().parents[1]


def test_memory_js_exposes_metadata_editors_and_patch() -> None:
    text = (HUB_ROOT / "static" / "js" / "memory.js").read_text(encoding="utf-8")
    for needle in (
        'method: "PATCH"',
        'selectField("Confidence"',
        'selectField("Priority"',
        'selectField("Provenance"',
        "data-vis=",
        "Save metadata",
        "always_inject",
    ):
        assert needle in text, f"missing {needle!r} in memory.js"


def test_template_includes_cards_recall_profiles() -> None:
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'value="biographical.v1"' in template
    assert 'value="self.factual.v1"' in template
    assert 'id="memorySubviewConsolidationDrafts"' in template
    assert 'id="memoryConsolidationDraftsPanel"' in template


def test_memory_js_includes_consolidation_drafts_subview() -> None:
    text = (HUB_ROOT / "static" / "js" / "memory.js").read_text(encoding="utf-8")
    assert "/api/memory/consolidation/drafts" in text
    assert "consolidation_draft_id" in text
    assert "memorySubviewConsolidationDrafts" in text
    assert "onRejected" in text
    assert "activeConsolidationDraftId === draftId" in text
