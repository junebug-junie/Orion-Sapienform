"""Which day is it, for Orion. One helper, pure.

``day_id`` is the local calendar date (America/Denver by default, the same zone and the
same half-open window as Orion's Day letter, ``orion/orion_day/window.py``), so the
chronology and the letter can never disagree about which day a row belongs to.

``day_phase_for`` mirrors ``orion/situational/context.py:_day_phase_label`` (the
``TimeContextV1.day_phase`` vocabulary). That module is too heavy to import from a pure
reducer, so the table is copied and ``test_day_boundary.py`` pins it minute-for-minute
against the original.

Timestamp casts, stated once (every source adapter goes through ``as_utc``):

* ``timestamptz`` columns arrive aware and are converted to UTC.
* Naive ``timestamp`` columns (``chat_history_log.created_at``,
  ``metacog_trigger.timestamp``) are written by servers whose clock and Postgres session
  are ``Etc/UTC`` (verified live 2026-10-10: ``SHOW timezone`` = ``Etc/UTC``; a chat row is
  stamped ~1 min after its cortex attention row's aware ``generated_at``). Naive is UTC.
* TEXT ISO columns (``orion_metacog.timestamp``) are parsed; naive text is UTC.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from orion.orion_day.window import orion_day_window
from orion.schemas.orion_day import ORION_DAY_TIMEZONE

DEFAULT_TZ = ORION_DAY_TIMEZONE


def as_utc(value: datetime | str | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def day_id_for(at: datetime, tz_name: str = DEFAULT_TZ) -> str:
    aware = as_utc(at)
    assert aware is not None
    return aware.astimezone(ZoneInfo(tz_name)).date().isoformat()


def day_window(day_id: str, tz_name: str = DEFAULT_TZ) -> tuple[datetime, datetime]:
    """Half-open UTC window [local midnight, next local midnight). DST-safe."""
    w = orion_day_window(date.fromisoformat(day_id), tz_name=tz_name)
    return w.window_start, w.window_end


def day_phase_for(at: datetime, tz_name: str = DEFAULT_TZ) -> str:
    aware = as_utc(at)
    assert aware is not None
    local = aware.astimezone(ZoneInfo(tz_name))
    hm = local.hour * 60 + local.minute
    if hm < 300:
        return "pre_dawn"
    if hm < 420:
        return "dawn"
    if hm < 720:
        return "morning"
    if hm < 840:
        return "midday"
    if hm < 1080:
        return "afternoon"
    if hm < 1200:
        return "dusk"
    return "night"
