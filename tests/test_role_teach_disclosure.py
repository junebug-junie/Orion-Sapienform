"""Unit tests for Mind → role teach disclosure formatter + splice."""

from __future__ import annotations

from orion.curiosity.role_teach_disclosure import (
    format_budget_spent_progress,
    format_role_teach_disclosure,
    splice_role_teach_disclosure,
)


def test_budget_spent_progress_names_resume_not_rehire() -> None:
    lines = format_budget_spent_progress(status="refused_budget", next_hop_n=3)
    text = "\n".join(lines).lower()
    assert "budget" in text
    assert "helprequest" in text.replace(" ", "") or "help request" in text
    assert "do not" in text or "don't" in text
    assert "resume" in text or "hop" in text
    assert format_budget_spent_progress(status="ok", next_hop_n=3) == []


def test_format_none_or_empty_returns_empty() -> None:
    assert format_role_teach_disclosure(None) == []
    assert format_role_teach_disclosure({}) == []


def test_format_all_unknown_and_empty_foresight_returns_empty() -> None:
    assert (
        format_role_teach_disclosure(
            {
                "expected_depth": "unknown",
                "cross_cutting": "unknown",
                "foresight_note": "",
                "noise_key": "ignored",
            }
        )
        == []
    )
    assert (
        format_role_teach_disclosure(
            {"expected_depth": "unknown", "cross_cutting": "unknown"}
        )
        == []
    )


def test_format_deep_includes_strong_hire_cursor_nudge() -> None:
    lines = format_role_teach_disclosure(
        {
            "expected_depth": "deep",
            "cross_cutting": "yes",
            "foresight_note": "Lease TTL archaeology.",
        }
    )
    text = "\n".join(lines).lower()
    assert "hire_cursor" in text or "hire cursor" in text
    assert "strongly" in text or "strong" in text
    assert "expensive" not in text


def test_format_present_shape_returns_advisory_lines() -> None:
    lines = format_role_teach_disclosure(
        {
            "expected_depth": "deep",
            "cross_cutting": "yes",
            "foresight_note": "Likely multi-service archaeology.",
            "user_intent": "Trace substrate.route edges",
            "attention_frontier": [{"label": "x"}],  # not allow-listed
        }
    )
    text = "\n".join(lines)
    assert lines
    assert "advisory" in text.lower()
    assert "deep" in text
    assert "yes" in text or "cross" in text.lower()
    assert "multi-service archaeology" in text
    assert "Trace substrate.route edges" in text
    assert "attention_frontier" not in text


def test_format_foresight_alone_with_unknown_labels_still_discloses() -> None:
    lines = format_role_teach_disclosure(
        {
            "expected_depth": "unknown",
            "cross_cutting": "unknown",
            "foresight_note": "This may touch ACL grants",
        }
    )
    assert lines
    assert any("ACL grants" in line for line in lines)


def test_format_user_intent_alone_with_unknown_labels_still_discloses() -> None:
    """Non-empty user_intent must keep disclosure when depth/cross/foresight empty."""
    lines = format_role_teach_disclosure(
        {
            "expected_depth": "unknown",
            "cross_cutting": "unknown",
            "foresight_note": "",
            "user_intent": "Trace substrate.route edges",
        }
    )
    assert lines
    assert any("Trace substrate.route edges" in line for line in lines)
    assert any(line.startswith("- intent:") for line in lines)


def test_format_foresight_unknown_literal_omitted_like_empty() -> None:
    assert (
        format_role_teach_disclosure(
            {
                "expected_depth": "unknown",
                "cross_cutting": "unknown",
                "foresight_note": "unknown",
            }
        )
        == []
    )
    lines = format_role_teach_disclosure(
        {
            "expected_depth": "deep",
            "cross_cutting": "unknown",
            "foresight_note": "unknown",
        }
    )
    assert lines
    assert "deep" in "\n".join(lines)
    assert not any("foresight:" in line for line in lines)


def test_format_appends_progress_lines_and_strips_blanks() -> None:
    lines = format_role_teach_disclosure(
        {"expected_depth": "shallow"},
        progress_lines=("  hop 2 already wrote a prior  ", "", "  ", "next: look at acl"),
    )
    text = "\n".join(lines)
    assert "shallow" in text
    assert "hop 2 already wrote a prior" in text
    assert "next: look at acl" in text
    assert "" not in lines  # blanks stripped from progress


def test_format_progress_alone_discloses_even_when_shape_unknown() -> None:
    lines = format_role_teach_disclosure(
        {"expected_depth": "unknown", "cross_cutting": "unknown"},
        progress_lines=("Already wrote hop 1 notes.",),
    )
    assert lines
    assert any("hop 1" in line for line in lines)


def test_splice_empty_extra_is_identity() -> None:
    prompt = "hello\nASKING FOR CONTRACTOR HELP. Write a HelpRequest\nbye"
    assert splice_role_teach_disclosure(prompt, ()) == prompt
    assert splice_role_teach_disclosure(prompt, []) == prompt


def test_splice_inserts_before_contractor_help_marker() -> None:
    prompt = (
        "YOUR ROLE FOR THIS SITTING.\n"
        "\n"
        "ASKING FOR CONTRACTOR HELP. Write a HelpRequest only when hiring.\n"
        "more\n"
    )
    extra = [
        "Mind work-shape for this sitting (advisory):",
        "- expected depth: deep",
    ]
    out = splice_role_teach_disclosure(prompt, extra)
    assert "ASKING FOR CONTRACTOR HELP" in out
    role_idx = out.index("YOUR ROLE FOR THIS SITTING")
    disclosure_idx = out.index("Mind work-shape for this sitting")
    help_idx = out.index("ASKING FOR CONTRACTOR HELP")
    assert role_idx < disclosure_idx < help_idx
    assert "expected depth: deep" in out


def test_splice_is_identity_when_role_teach_markers_missing() -> None:
    """No contractor-help / sitting-role markers → leave prompt unchanged."""
    prompt = "just a frozen kickoff without role section\n"
    extra = ["Mind work-shape for this sitting (advisory):", "- foresight: ACL"]
    assert splice_role_teach_disclosure(prompt, extra) == prompt


def test_splice_is_idempotent() -> None:
    prompt = (
        "YOUR ROLE FOR THIS SITTING.\n"
        "ASKING FOR CONTRACTOR HELP. Write a HelpRequest.\n"
    )
    extra = [
        "Mind work-shape for this sitting (advisory):",
        "- expected depth: deep",
    ]
    once = splice_role_teach_disclosure(prompt, extra)
    twice = splice_role_teach_disclosure(once, extra)
    assert once == twice
    assert once.count("Mind work-shape for this sitting (advisory):") == 1
