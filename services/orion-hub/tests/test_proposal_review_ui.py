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
    assert "Rationale:" in ui
    assert "Evidence:" in ui
    assert "Risk:" in ui
    assert "Confidence:" in ui
    assert "mutation_allowed=" in ui
    assert "requires_human_approval=" in ui
    ui_lower = ui.lower()
    for forbidden in ("/triage", "/review", 'method="post"', "request-changes"):
        assert forbidden not in ui_lower
    for forbidden_button in ('id="approve', 'id="reject', ">approve<", ">reject<", "approve proposal", "reject proposal"):
        assert forbidden_button not in ui_lower
