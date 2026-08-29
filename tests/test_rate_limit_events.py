"""Structural detection of real rate-limit events, and the reset time they carry.

Fixtures are synthetic transcripts in tmp_path, not the live corpus, so these
do not drift with whatever is on this machine today.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.dev_economics.rate_limit_events import (
    classify,
    observe,
    parse_reset_at,
    was_rate_limited_recently,
)

T = datetime(2026, 8, 14, 18, 42, tzinfo=timezone.utc)


def line(ts: datetime, *, text: str | None = None, api_error: bool = False, kind: str = "assistant"):
    obj: dict = {"type": kind, "timestamp": ts.isoformat().replace("+00:00", "Z")}
    if api_error:
        obj["isApiErrorMessage"] = True
    if text is not None:
        obj["message"] = {"content": [{"type": "text", "text": text}]}
    return json.dumps(obj)


def transcript(tmp_path, *lines):
    d = tmp_path / "-some-project"
    d.mkdir(exist_ok=True)
    p = d / "session.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return tmp_path


# --------------------------------------------------------------------------
# Detection is structural. The first version of this module counted 12 events
# in the live 5h window and every one was this session's own tool output
# discussing rate limits -- a detector measuring the investigation that
# produced it.
# --------------------------------------------------------------------------


def test_a_real_session_limit_is_a_limit_event():
    obj = json.loads(line(T, text="You've hit your session limit · resets 8pm (UTC)", api_error=True))
    result = classify(obj)
    assert result is not None and result[0] == "session_limit"


def test_a_real_weekly_limit_is_distinguished():
    obj = json.loads(line(T, text="You've hit your weekly limit · resets Aug 18, 3am (UTC)", api_error=True))
    result = classify(obj)
    assert result is not None and result[0] == "weekly_limit"


@pytest.mark.parametrize("text", [
    "Failed to authenticate. API Error: 401 Invalid bearer token",
    "Prompt is too long",
    "Your organization has disabled Claude subscription access for this account",
])
def test_other_api_errors_are_not_scarcity(text):
    """All three carry isApiErrorMessage and none of them is a rate limit."""
    assert classify(json.loads(line(T, text=text, api_error=True))) is None


def test_prose_about_rate_limits_is_not_an_event():
    """The bug this pins: a transcript line quoting `rate_limit_error` from a
    grep, with no isApiErrorMessage flag, was counted as a real limit.
    """
    prose = "  5 files : rate_limit_error\n  1 files : usage limit reached"
    assert classify(json.loads(line(T, text=prose))) is None
    assert classify(json.loads(line(T, text="You've hit your session limit"))) is None


# --------------------------------------------------------------------------
# Reset times, hand-computed.
# --------------------------------------------------------------------------


def test_reset_time_same_day():
    """Event 18:42, 'resets 8pm' -> 20:00 the same day."""
    at = parse_reset_at("You've hit your session limit · resets 8pm (UTC)", T)
    assert at == datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc)


def test_reset_time_with_minutes():
    at = parse_reset_at("resets 3:30am (UTC)", datetime(2026, 8, 14, 1, 0, tzinfo=timezone.utc))
    assert at == datetime(2026, 8, 14, 3, 30, tzinfo=timezone.utc)


def test_reset_time_rolls_to_tomorrow_when_already_past():
    """Event at 05:00, 'resets 3:30am' -- that hour is gone today."""
    at = parse_reset_at("resets 3:30am (UTC)", datetime(2026, 8, 14, 5, 0, tzinfo=timezone.utc))
    assert at == datetime(2026, 8, 15, 3, 30, tzinfo=timezone.utc)


def test_reset_time_noon_and_midnight_meridiem():
    base = datetime(2026, 8, 14, 1, 0, tzinfo=timezone.utc)
    assert parse_reset_at("resets 12pm (UTC)", base) == datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    assert parse_reset_at("resets 12am (UTC)", base) == datetime(2026, 8, 15, 0, 0, tzinfo=timezone.utc)


def test_weekly_reset_carries_a_date():
    at = parse_reset_at("resets Aug 18, 3am (UTC)", T)
    assert at == datetime(2026, 8, 18, 3, 0, tzinfo=timezone.utc)


def test_no_reset_time_is_none_not_a_guess():
    assert parse_reset_at("You've hit your session limit", T) is None


# --------------------------------------------------------------------------
# State.
# --------------------------------------------------------------------------


def test_limited_until_the_stated_reset(tmp_path):
    root = transcript(tmp_path, line(T, text="You've hit your session limit · resets 8pm (UTC)", api_error=True))
    o = observe(now=T + timedelta(minutes=18), window_hours=1.0, root=root)

    assert o.state == "limited"
    assert o.resets_at == datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc)
    assert o.seconds_until_reset == pytest.approx(3600.0)  # 19:00 -> 20:00
    assert was_rate_limited_recently(1.0, now=T + timedelta(minutes=18), root=root) is True


def test_clear_once_the_stated_reset_has_passed(tmp_path):
    root = transcript(tmp_path, line(T, text="You've hit your session limit · resets 8pm (UTC)", api_error=True))
    o = observe(now=datetime(2026, 8, 14, 20, 30, tzinfo=timezone.utc), window_hours=2.0, root=root)

    assert o.state == "clear"
    assert o.resets_at is None
    assert o.event_count == 1  # still counted as pressure


def test_the_stated_reset_beats_the_activity_heuristic(tmp_path):
    """Live bug: activity at 18:50 reported `clear` an hour inside a limit
    stated to hold until 20:00. Activity after a limit is often failed retries.
    """
    root = transcript(
        tmp_path,
        line(T, text="You've hit your session limit · resets 8pm (UTC)", api_error=True),
        line(T + timedelta(minutes=8), text="retrying"),
    )
    o = observe(now=T + timedelta(minutes=18), window_hours=1.0, root=root)
    assert o.state == "limited"


def test_activity_fallback_only_when_no_reset_time(tmp_path):
    root = transcript(
        tmp_path,
        line(T, text="You've hit your session limit", api_error=True),
        line(T + timedelta(minutes=8), text="a real answer"),
    )
    assert observe(now=T + timedelta(minutes=18), window_hours=1.0, root=root).state == "clear"


def test_failed_retries_do_not_count_as_recovery(tmp_path):
    """No reset time, so the fallback applies -- but every later message is
    itself an API error, which is not evidence the limit lifted.
    """
    root = transcript(
        tmp_path,
        line(T, text="You've hit your session limit", api_error=True),
        line(T + timedelta(minutes=4), text="You've hit your session limit", api_error=True),
        line(T + timedelta(minutes=8), text="You've hit your session limit", api_error=True),
    )
    o = observe(now=T + timedelta(minutes=18), window_hours=1.0, root=root)
    assert o.state == "limited"
    assert o.event_count == 3


def test_no_events_with_activity_is_clear(tmp_path):
    root = transcript(tmp_path, line(T, text="ordinary turn"))
    o = observe(now=T + timedelta(minutes=10), window_hours=1.0, root=root)
    assert o.state == "clear"
    assert was_rate_limited_recently(1.0, now=T + timedelta(minutes=10), root=root) is False


def test_nothing_observed_is_unknown_not_clear(tmp_path):
    """The distinction that matters: an unobservable window is not an empty
    road. Same contract as quota_budget.WindowSpend.observed.
    """
    root = transcript(tmp_path, line(T - timedelta(days=3), text="old turn"))
    o = observe(now=T, window_hours=1.0, root=root)

    assert o.observed is False
    assert o.state == "unknown"
    assert o.staleness_sec is None
    assert was_rate_limited_recently(1.0, now=T, root=root) is None


def test_missing_root_is_unknown(tmp_path):
    o = observe(now=T, window_hours=1.0, root=tmp_path / "does-not-exist")
    assert o.state == "unknown"
    assert o.scanned_file_count == 0


def test_events_outside_the_window_are_not_counted(tmp_path):
    root = transcript(
        tmp_path,
        line(T - timedelta(hours=6), text="You've hit your session limit · resets 8pm (UTC)", api_error=True),
        line(T, text="ordinary turn"),
    )
    o = observe(now=T + timedelta(minutes=5), window_hours=1.0, root=root)
    assert o.event_count == 0
    assert o.state == "clear"


def test_staleness_is_measured_from_window_end(tmp_path):
    root = transcript(tmp_path, line(T, text="ordinary turn"))
    o = observe(now=T + timedelta(minutes=30), window_hours=1.0, root=root)
    assert o.staleness_sec == pytest.approx(1800.0)
