from __future__ import annotations

from orion.discussion_window.timeframe import parse_journal_discussion_lookback_seconds


def test_parse_minutes_variants() -> None:
    assert parse_journal_discussion_lookback_seconds("Journal the last 34 minutes") == 34 * 60
    assert parse_journal_discussion_lookback_seconds("please journal past 5 mins") == 5 * 60
    assert parse_journal_discussion_lookback_seconds("journal for the last 1 minute") == 60


def test_parse_hours_variants() -> None:
    assert parse_journal_discussion_lookback_seconds("Journal the last hour") == 3600
    assert parse_journal_discussion_lookback_seconds("journal for the last hour") == 3600
    assert parse_journal_discussion_lookback_seconds("journal last 2 hours") == 2 * 3600


def test_parse_day_variants() -> None:
    assert parse_journal_discussion_lookback_seconds("journal our chat discussion for the last day") == 86400
    assert parse_journal_discussion_lookback_seconds("journal last 24 hours") == 86400
    assert parse_journal_discussion_lookback_seconds("journal last 3 days") == 3 * 86400


def test_parse_chat_discussion_default_day() -> None:
    assert parse_journal_discussion_lookback_seconds("journal our chat discussion") == 86400


def test_parse_non_journal_returns_none() -> None:
    assert parse_journal_discussion_lookback_seconds("what happened in the last hour") is None
