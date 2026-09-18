"""Investigation subject: short self-authored text for Mind, not the kickoff."""

from __future__ import annotations

from orion.curiosity.investigation_subject import build_investigation_subject


def test_subject_prefers_claim_and_continue_note() -> None:
    text = build_investigation_subject(
        claim="Concept decay never reduces activation",
        continue_note="Still do not know who sets half-life",
    )
    assert "Concept decay never reduces activation" in text
    assert "half-life" in text
    assert "MERGE (h:HelpRequest" not in text
    assert "ASKING FOR CONTRACTOR" not in text


def test_subject_when_claim_missing() -> None:
    text = build_investigation_subject(claim=None, continue_note="keep pulling on ACL")
    assert "not yet chosen" in text.lower() or "no claim" in text.lower()
    assert "ACL" in text


def test_subject_clips_long_inputs() -> None:
    huge = "x" * 5000
    text = build_investigation_subject(claim=huge, continue_note=huge, max_chars=400)
    assert len(text) <= 400
