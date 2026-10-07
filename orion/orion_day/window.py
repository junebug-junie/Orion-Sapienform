"""The letter's day: one America/Denver calendar day, as a half-open UTC window.

Same day as the compactors' "day" mode (orion/cognition/compactor/calendar_day.py
``previous_local_day_window``, used by resolve_chat_compactor_window): the
calendar_date matches exactly. The bounds differ only in form -- the compactors
end at 23:59:59.999999 inclusive, this ends at the next local midnight exclusive
-- so a row stamped in the last microsecond is not lost. DST-safe: a 23 h or 25 h
day is still one day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from orion.schemas.orion_day import ORION_DAY_TIMEZONE


@dataclass(frozen=True)
class OrionDayWindow:
    letter_date: date
    window_start: datetime  # UTC, inclusive (local 00:00)
    window_end: datetime  # UTC, exclusive (next local 00:00)
    timezone_name: str


def orion_day_window(letter_date: date, *, tz_name: str = ORION_DAY_TIMEZONE) -> OrionDayWindow:
    tz = ZoneInfo(tz_name)
    start = datetime.combine(letter_date, time.min, tzinfo=tz)
    end = datetime.combine(letter_date + timedelta(days=1), time.min, tzinfo=tz)
    return OrionDayWindow(
        letter_date=letter_date,
        window_start=start.astimezone(timezone.utc),
        window_end=end.astimezone(timezone.utc),
        timezone_name=tz_name,
    )


def yesterday_letter_date(now: datetime, *, tz_name: str = ORION_DAY_TIMEZONE) -> date:
    aware = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return aware.astimezone(ZoneInfo(tz_name)).date() - timedelta(days=1)
