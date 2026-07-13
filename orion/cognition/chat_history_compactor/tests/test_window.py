from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from orion.cognition.chat_history_compactor.window import (
    ResolvedChatCompactorWindow,
    resolve_chat_compactor_window,
)


def test_resolve_day_window_uses_yesterday_denver() -> None:
    now = datetime(2026, 7, 9, 6, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode="day",
        lookback_hours=None,
        now=now,
        user_text="",
        workflow_request={"window_mode": "day"},
    )
    assert isinstance(resolved, ResolvedChatCompactorWindow)
    assert resolved.mode == "day"
    assert resolved.calendar_date == "2026-07-08"
    assert resolved.compactor_index == "chat_compactor:day:2026-07-08"
    assert resolved.lookback_seconds == 86400
    assert resolved.window_end.astimezone(ZoneInfo("America/Denver")).date().isoformat() == "2026-07-08"


def test_resolve_rolling_from_prompt_six_hours() -> None:
    now = datetime(2026, 7, 9, 10, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode=None,
        lookback_hours=None,
        now=now,
        user_text="Compact the last 6 hours of chat into a memory digest.",
        workflow_request={},
    )
    assert resolved.mode == "rolling"
    assert resolved.lookback_hours == 6
    assert resolved.compactor_index.startswith("chat_compactor:rolling:6h:")


def test_day_window_covers_end_of_day_microseconds() -> None:
    now = datetime(2026, 7, 9, 6, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode="day",
        lookback_hours=None,
        now=now,
        user_text="",
        workflow_request={},
    )
    end_local = resolved.window_end.astimezone(ZoneInfo("America/Denver"))
    assert end_local.timetz().replace(tzinfo=None) == time.max


def test_request_lookback_hours_capped_at_fourteen_days() -> None:
    now = datetime(2026, 7, 9, 10, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode=None,
        lookback_hours=None,
        now=now,
        user_text="",
        workflow_request={"lookback_hours": 720},
    )
    assert resolved.lookback_hours == 24 * 14
    assert resolved.lookback_seconds == 24 * 14 * 3600


def test_unknown_window_mode_fails_loud() -> None:
    now = datetime(2026, 7, 9, 10, 0, tzinfo=ZoneInfo("America/Denver"))
    with pytest.raises(ValueError, match="unsupported_chat_compactor_window_mode:daily"):
        resolve_chat_compactor_window(
            window_mode="daily",
            lookback_hours=None,
            now=now,
            user_text="",
            workflow_request={},
        )
