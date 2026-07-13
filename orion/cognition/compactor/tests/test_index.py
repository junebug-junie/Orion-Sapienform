from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from orion.cognition.compactor.index import build_compactor_index


def test_build_compactor_index_day() -> None:
    key = build_compactor_index(
        kind="chat_history_log",
        mode="day",
        calendar_date="2026-07-08",
    )
    assert key == "chat_compactor:day:2026-07-08"


def test_build_compactor_index_rolling_floors_to_minute() -> None:
    start = datetime(2026, 7, 9, 4, 0, 37, tzinfo=ZoneInfo("America/Denver"))
    key = build_compactor_index(
        kind="chat_history_log",
        mode="rolling",
        lookback_hours=6,
        window_start=start,
    )
    assert key == "chat_compactor:rolling:6h:2026-07-09T04:00:00-06:00"


def test_build_compactor_index_day_requires_calendar_date() -> None:
    with pytest.raises(ValueError, match="calendar_date"):
        build_compactor_index(kind="chat_history_log", mode="day")


def test_build_compactor_index_rolling_requires_hours_and_start() -> None:
    with pytest.raises(ValueError, match="lookback_hours"):
        build_compactor_index(
            kind="chat_history_log",
            mode="rolling",
            window_start=datetime(2026, 7, 9, 4, 0, tzinfo=ZoneInfo("America/Denver")),
        )
