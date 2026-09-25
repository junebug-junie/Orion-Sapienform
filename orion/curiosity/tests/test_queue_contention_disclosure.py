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


def test_queue_line_prefers_hire_not_compete_with_gpu() -> None:
    text = "\n".join(format_queue_contention_progress(score=8.0, driver=SOURCE_DURABLE)).lower()
    assert "hire_cursor" in text or "hire cursor" in text
    assert "compete" not in text
    assert "prefer hire when mind says deep" not in text
    # Elevated queue must push hire, never excuse staying local.
    assert "reason to hire" in text or "reason to hire_cursor" in text
    assert "not a reason to stay" in text or "not a reason to" in text
    assert "no hire" not in text
    assert "uneconom" not in text
    # No short-look theater — hire now (ban positive "short look"; allow "do not take a short").
    assert "short tried_summary" not in text
    assert "keep only a short" not in text
    assert "after a short" not in text
    assert "do not take a short local look" in text
    assert "quick look" not in text
    assert "helprequest" in text.replace(" ", "") or "help request" in text


def test_oldest_wait_driver_gets_stuck_wording_without_raw_age() -> None:
    from orion.field.queue_contention import OLDEST_WAIT_SUFFIX, SOURCE_SEED

    lines = format_queue_contention_progress(10.0, SOURCE_SEED + OLDEST_WAIT_SUFFIX)
    assert len(lines) == 1
    line = lines[0]
    assert "10/10 (high)" in line
    assert "oldest reading seed has waited far longer" in line
    assert "stuck" in line
    # Normalized only: no hours/days/seconds or raw counts reach Orion.
    assert not any(ch.isdigit() for ch in line.replace("10/10", ""))


def test_every_oldest_wait_driver_has_its_own_blurb() -> None:
    from orion.field.queue_contention import OLDEST_WAIT_SUFFIX, SOURCE_KEYS

    generic = "shared agent capacity is under more contention than usual"
    for src in SOURCE_KEYS:
        (line,) = format_queue_contention_progress(6.0, src + OLDEST_WAIT_SUFFIX)
        assert generic not in line
        assert "oldest" in line


def test_stuck_seed_queue_does_not_push_a_hire() -> None:
    """A frozen seed queue is a stalled pipeline, not capacity: no hire-now nudge."""
    from orion.field.queue_contention import OLDEST_WAIT_SUFFIX, SOURCE_SEED

    (line,) = format_queue_contention_progress(10.0, SOURCE_SEED + OLDEST_WAIT_SUFFIX)
    assert "Write hire_cursor" not in line
    assert "not by itself a reason to hire_cursor" in line


def test_capacity_oldest_wait_drivers_keep_the_hire_nudge() -> None:
    from orion.field.queue_contention import OLDEST_WAIT_SUFFIX, SOURCE_GPU_POOL, SOURCE_SEED

    for driver in (SOURCE_DURABLE + OLDEST_WAIT_SUFFIX, SOURCE_GPU_POOL + OLDEST_WAIT_SUFFIX, SOURCE_SEED):
        (line,) = format_queue_contention_progress(6.0, driver)
        assert "Write hire_cursor and HelpRequest now" in line
