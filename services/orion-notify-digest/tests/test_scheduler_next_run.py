"""Regression: _next_run_utc() must return a naive datetime.

Found live 2026-07-28: this function used to return a tz-aware datetime while every
other datetime in this file (datetime.utcnow()) is naive -- _scheduler_loop's very
first iteration crashed on every single startup with "can't subtract offset-naive and
offset-aware datetimes", silently, because nothing retrieved the exception from the
fire-and-forget asyncio task running the loop. The scheduler had never once completed
an iteration since this service's inception.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import app.main as digest_main


def test_next_run_utc_returns_naive_datetime() -> None:
    now = datetime(2026, 7, 28, 9, 0, 0)  # naive, matches datetime.utcnow()'s shape
    result = digest_main._next_run_utc(now, "07:30")
    assert result.tzinfo is None


def test_next_run_utc_subtractable_from_naive_utcnow() -> None:
    """The exact operation that crashed live: (next_run - now).total_seconds()."""
    now = datetime(2026, 7, 28, 9, 0, 0)
    next_run = digest_main._next_run_utc(now, "07:30")
    sleep_for = (next_run - now).total_seconds()
    assert sleep_for >= 0


def test_next_run_utc_accepts_aware_input_too() -> None:
    """The function must not assume its input is naive -- explicit UTC tzinfo must
    not break it either."""
    now_naive = datetime(2026, 7, 28, 9, 0, 0)
    now_aware = now_naive.replace(tzinfo=timezone.utc)
    naive_result = digest_main._next_run_utc(now_naive, "07:30")
    aware_result = digest_main._next_run_utc(now_aware, "07:30")
    assert naive_result == aware_result


def test_next_run_utc_schedules_next_day_when_time_already_passed() -> None:
    # 09:00 UTC on 2026-07-28 = 03:00 MDT -- 07:30 MDT hasn't happened yet today.
    now = datetime(2026, 7, 28, 9, 0, 0)
    next_run = digest_main._next_run_utc(now, "07:30")
    assert next_run > now
    assert (next_run - now) < timedelta(days=1, hours=1)


def test_next_run_utc_schedules_same_day_when_time_not_yet_passed() -> None:
    # 06:00 UTC on 2026-07-28 = 00:00 MDT -- 07:30 MDT is still later today.
    now = datetime(2026, 7, 28, 6, 0, 0)
    next_run = digest_main._next_run_utc(now, "07:30")
    assert next_run.date() == now.date()
