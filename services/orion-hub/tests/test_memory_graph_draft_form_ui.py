from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FORM_JS = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "memory-graph-draft-form.js"


def test_memory_js_approve_sends_card_projection_defaults() -> None:
    text = (REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "memory.js").read_text(encoding="utf-8")
    assert "card_projection_defaults" in text
    assert "buildCardProjectionPayload" in text
    text = FORM_JS.read_text(encoding="utf-8")
    for needle in (
        "Card metadata (applied on Approve",
        "buildCardProjectionPayload",
        "Juniper",
        'type = "date"',
        "entitySelect",
        "renderParticipantRows",
        "Holder (who feels)",
        "Occurred (local date)",
        "data-card-meta-root",
    ):
        assert needle in text, f"missing {needle!r} in memory-graph-draft-form.js"
