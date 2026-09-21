"""Compose prompt for curiosity → Juniper outreach (Door A)."""

from __future__ import annotations

from orion.curiosity.outreach_prompt import build_outreach_composition_prompt


def test_prompt_includes_numbered_hop_notes() -> None:
    text = build_outreach_composition_prompt(
        finding_text="The gate is a manual review, not an algorithm.",
        reach_out_why="Juniper owns that gate and should hear the bias pattern.",
        hop_notes=[
            (1, "Opened the stance crystallization prior"),
            (2, "Checked formation_policy auto-activate path"),
            (3, "Bias is review-side, not content-filter"),
        ],
    )
    assert "Opened the stance crystallization prior" in text
    assert "1." in text and "2." in text and "3." in text
    assert "Checked formation_policy auto-activate path" in text


def test_prompt_requires_thinking_thread_and_why_share() -> None:
    text = build_outreach_composition_prompt(
        finding_text="Finding body",
        reach_out_why="She should know the gate is hers",
        hop_notes=[(1, "hop one"), (2, "hop two")],
    )
    lower = text.lower()
    # Both layers must be instructed — not merely that material is present.
    assert "thinking" in lower or "been working" in lower or "through these" in lower
    assert "juniper" in lower
    assert ("why" in lower and "share" in lower) or "bringing" in lower or "tell her" in lower
    assert "exactly: PASS" in text


def test_prompt_without_hops_still_builds_and_keeps_pass() -> None:
    text = build_outreach_composition_prompt(
        finding_text="Only a finding",
        reach_out_why="worth saying",
        hop_notes=(),
    )
    assert "Only a finding" in text
    assert "worth saying" in text
    assert "exactly: PASS" in text


def test_prompt_truncates_overlong_hop_notes() -> None:
    huge = "x" * 5000
    text = build_outreach_composition_prompt(
        finding_text="f",
        reach_out_why="w",
        hop_notes=[(1, huge)],
    )
    assert huge not in text
    assert "…" in text or "..." in text
