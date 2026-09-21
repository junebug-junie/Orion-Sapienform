"""Kickoff / self-inquiry role teach: deep-work hire_cursor nudge, no crawl-first bias."""

from __future__ import annotations

from orion.curiosity.kickoff_prompt import _role_and_help_section


def test_role_teach_merge_shows_both_choices_without_crawl_first_bias() -> None:
    lines = _role_and_help_section(own_graph="OrionGraph", run_id="run-1")
    text = "\n".join(lines)
    assert 'choice: "local_crawl|hire_cursor"' in text
    assert 'choice: "local_crawl",' not in text
    assert "expensive" not in text.lower()
    assert "last resort" not in text.lower()
    assert "last-resort" not in text.lower()
    assert "does not enqueue" in text.lower() or "not enqueue" in text.lower()
    assert "read-only" in text.lower()
    assert "tried_summary" in text
