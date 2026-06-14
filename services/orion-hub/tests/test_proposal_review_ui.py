from __future__ import annotations

from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]


def test_proposal_review_ui_wired() -> None:
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    ui = (HUB_ROOT / "static" / "js" / "proposal-review-ui.js").read_text(encoding="utf-8")

    assert "proposal-review-ui.js" in template
    assert 'id="proposalReviewPanel"' in template
    assert "Pending Decisions" in template
    assert "/api/proposal-review/pending" in ui
    assert "/api/proposal-review/proposals/" in ui
    assert "No pending decisions." in ui
    assert "Proposal review API unavailable." in ui
    assert "Current belief:" in ui
    assert "Proposed correction:" in ui
    assert "mutation_allowed=" in ui
    assert "requires_human_approval=" in ui
    assert "approve" not in ui.lower() or "approval inbox" not in ui.lower()
