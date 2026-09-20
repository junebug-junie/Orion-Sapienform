"""Queue contention progress lines for hire role-teach (read-only score)."""

from __future__ import annotations

from orion.curiosity.queue_contention_disclosure import format_queue_contention_progress
from orion.field.queue_contention import SOURCE_DURABLE


def test_queue_line_names_score_and_driver_not_raw_count() -> None:
    lines = format_queue_contention_progress(score=8.0, driver=SOURCE_DURABLE)
    text = "\n".join(lines)
    assert "8" in text and "/10" in text
    assert "durable" in text.lower() or "demand" in text.lower()
    assert "pending" not in text.lower() or "121" not in text  # no raw backlog dump


def test_queue_line_omits_when_score_none_or_nonpositive() -> None:
    assert format_queue_contention_progress(score=None, driver=SOURCE_DURABLE) == []
    assert format_queue_contention_progress(score=0.0, driver=SOURCE_DURABLE) == []
    assert format_queue_contention_progress(score=-1.0, driver=None) == []


def test_queue_line_omits_fractional_score_that_would_read_zero() -> None:
    """0.3 rounds to 0/10 but used to claim (elevated) — omit instead."""
    assert format_queue_contention_progress(score=0.3, driver=SOURCE_DURABLE) == []


def test_queue_line_mid_score_still_discloses() -> None:
    lines = format_queue_contention_progress(score=5.5, driver=SOURCE_DURABLE)
    text = "\n".join(lines)
    assert lines
    assert "/10" in text
    assert "elevated" in text or "moderate" in text or "high" in text
    # Shown integer must not be 0 while a band is claimed.
    assert "0/10" not in text


def test_queue_line_never_embeds_raw_counts() -> None:
    lines = format_queue_contention_progress(score=5.5, driver="world_pulse_seed_pending")
    text = "\n".join(lines).lower()
    assert "121" not in text
    assert "pending=" not in text
    assert "count" not in text
