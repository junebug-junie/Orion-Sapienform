from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

DEFAULT_COMPACTOR_TIMEZONE = "America/Denver"


@dataclass(frozen=True)
class CalendarDayWindow:
    calendar_date: str  # local date, ISO (YYYY-MM-DD)
    window_start: datetime  # UTC, inclusive (00:00:00 local)
    window_end: datetime  # UTC, inclusive (23:59:59.999999 local)
    timezone_name: str


def previous_local_day_window(now: datetime, *, tz_name: str = DEFAULT_COMPACTOR_TIMEZONE) -> CalendarDayWindow:
    """The full previous local calendar day relative to `now`.

    Shared by the chat and GitHub compactors so a scheduled run of either covers
    exactly the same Denver day (DST-correct: a 23h/25h day stays one day).
    """
    tz = ZoneInfo(tz_name)
    aware = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    yesterday = aware.astimezone(tz).date() - timedelta(days=1)
    start_local = datetime.combine(yesterday, time.min, tzinfo=tz)
    end_local = datetime.combine(yesterday, time.max, tzinfo=tz)
    return CalendarDayWindow(
        calendar_date=yesterday.isoformat(),
        window_start=start_local.astimezone(timezone.utc),
        window_end=end_local.astimezone(timezone.utc),
        timezone_name=tz_name,
    )
