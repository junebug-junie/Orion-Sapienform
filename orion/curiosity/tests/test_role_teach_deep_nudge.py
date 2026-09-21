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
    # Queue weather must not be teachable as anti-hire.
    lower = text.lower()
    assert "more reason to hire_cursor" in lower
    assert "never a reason to stay local" in lower
    assert "queue elevated so no hire" in lower  # named ban of the bad why
    assert "never local_crawl" not in lower  # do not hard-ban local for mild queue


def test_review_role_queue_elevated_means_hire_not_self() -> None:
    from orion.curiosity.kickoff_prompt import _review_role_section

    lower = "\n".join(_review_role_section(run_id="abc123")).lower()
    assert "prefer hire_cursor_review" in lower
    assert "prefer self_review, not hire_cursor_review" not in lower
    assert "backed up, that's a reason to prefer self_review" not in lower
