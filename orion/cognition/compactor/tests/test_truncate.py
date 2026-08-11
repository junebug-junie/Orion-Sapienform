from __future__ import annotations

from orion.cognition.compactor.truncate import truncate_at_word_boundary


def test_truncate_at_word_boundary_breaks_on_whitespace_not_mid_word() -> None:
    text = "one two three four five " + ("x" * 50)
    result, was_truncated = truncate_at_word_boundary(text, max_chars=20)
    assert was_truncated is True
    assert result.endswith("…")
    assert result == "one two three four…"


def test_truncate_at_word_boundary_breaks_on_newline_not_mid_token() -> None:
    # Real long-form text (bullet lists, numbered steps, markdown PR bodies)
    # is newline-delimited, not space-delimited -- a space-only boundary
    # check would still cut this mid-token.
    text = "\n".join(f"item{i}" for i in range(400))
    result, was_truncated = truncate_at_word_boundary(text, max_chars=1200)
    assert was_truncated is True
    assert result.endswith("…")
    trimmed = result[:-1]
    assert trimmed == text[: len(trimmed)]
    last_line = trimmed.rsplit("\n", 1)[-1]
    assert last_line.startswith("item") and last_line[4:].isdigit()  # not a partial token


def test_truncate_at_word_boundary_hard_cuts_single_unbroken_token() -> None:
    text = "x" * 2500
    result, was_truncated = truncate_at_word_boundary(text, max_chars=2000)
    assert was_truncated is True
    assert len(result) == 2001  # +ellipsis


def test_truncate_at_word_boundary_noop_when_within_budget() -> None:
    text = "well within budget"
    result, was_truncated = truncate_at_word_boundary(text, max_chars=100)
    assert was_truncated is False
    assert result == text
