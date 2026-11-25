"""Time windows in a chat message, for lived-memory recall (issue #3674).

"What happened with Amanda over the last month or so?" names a time. Recall used to ignore it:
older rows that fit the topic better filled the candidate pool, and the record from the named
month never reached the rerank (issue #3881). This module turns the time words in a message
into one window, ``[start, end)`` in UTC, resolved against "now" in Juniper's time zone
(``America/Denver`` by default).

**A small date grammar, not a word list.** A window is one of five shapes, tried in order:

1. a range of two dates ("November 2nd to 8th", "between Oct 3 and Oct 10", "the week of Nov 2");
2. one date ("October 30th", "Oct 18", "10/30", "2026-10-30", "the 14th of November");
3. a day relative to today ("yesterday", "the day before yesterday", "today", "last night");
4. a span relative to now:
   - ``the last/past N <unit>`` is a rolling span ending now ("over the last month or so");
   - ``N <unit> ago`` is a span centred on that point ("a month or so ago", "two weeks ago");
   - ``last/this <unit>`` starts at the beginning of the previous/current calendar unit and runs
     to now ("last week", "this month"), and ``last/this weekend`` is that Saturday and Sunday;
   - ``last/on <weekday>`` is the most recent such day;
5. a named month or year after a preposition ("in October", "early September", "since August",
   "in 2025"). A bare month name needs a preposition or a qualifier: "May I ask" is not May.

``N`` is a digit or a number word ("a", "a couple of", "a few", "two" .. "twelve"). A hedge
("or so", "about", "around", "roughly") widens a span by half. "since <X>" runs from X to now.

**Past state.** ``past_state`` is True when the window ends before today and the message asks
in the past tense with a state verb ("Where *was* I in early October?"). Recall then also reads
the as-of rows (``orion.memory.lived_memory.EPISODE_VALID_DURING_SQL``): facts true in the
window that an update has since retired. The query rewrite may also say so (``PAST: yes``).

Pure: the same text and ``now`` always give the same window. No model, no I/O.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

DEFAULT_TZ = "America/Denver"
# Memories are distilled after the conversation, and a "date" in a question is the day in the
# user's head: a record on the day before or after still counts as in the window.
WINDOW_PAD = timedelta(days=1)
# A model-written window wider than this, or older than MAX_AGE, is not a time the message
# named: it is dropped (the parser's own windows are never this wide except "last year").
MODEL_MAX_SPAN = timedelta(days=400)
MAX_AGE = timedelta(days=3 * 366)

SOURCE_PARSER = "parser"
SOURCE_MODEL = "model"

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5,
             "sunday": 6}
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "couple": 2, "few": 3, "several": 3}
_UNIT_DAYS = {"day": 1.0, "week": 7.0, "month": 30.44, "year": 365.25}

# Any month name, abbreviations allowed (only next to a day number, where they are unambiguous).
_MON = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
# Full month names only (a bare month needs a preposition anyway).
_MON_FULL = r"(?:" + "|".join(k for k in _MONTHS if len(k) > 4 or k in ("may", "june", "july")) + r")"
_DAY = r"(?:[12][0-9]|3[01]|0?[1-9])(?:st|nd|rd|th)?"
_YEAR = r"(?:20[0-9]{2})"
_NUM = r"(?:\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(?:a\s+)?couple(?:\s+of)?|(?:a\s+)?few|several)"
_UNIT = r"(?:day|week|month|year)s?"
_HEDGE_AFTER = r"(?:\s+or\s+so|\s+or\s+two|ish)?"
_HEDGE_BEFORE = re.compile(r"\b(?:about|around|roughly|maybe|like|almost|nearly|over|some)\s+$", re.I)

# One date: "October 30th", "Oct 30, 2026", "the 30th of October", "10/30", "2026-10-30".
def _date_re(p: str) -> str:
    return (rf"(?:(?P<{p}mon>{_MON})\s+(?P<{p}day>{_DAY})(?:,?\s+(?P<{p}year>{_YEAR}))?"
            rf"|(?:the\s+)?(?P<{p}day2>{_DAY})\s+of\s+(?P<{p}mon2>{_MON})(?:,?\s+(?P<{p}year2>{_YEAR}))?"
            rf"|(?P<{p}iso>{_YEAR}-[01][0-9]-[0-3][0-9])"
            rf"|(?P<{p}us>(?:1[0-2]|0?[1-9])/(?:[12][0-9]|3[01]|0?[1-9])(?:/(?:20)?[0-9]{{2}})?))")


_RANGE_RE = re.compile(
    rf"\b(?:between\s+|from\s+)?{_date_re('a')}\s*(?:to|through|thru|until|till|and|-|–|—)\s*"
    rf"(?:the\s+)?(?:(?P<bmon>{_MON})\s+)?(?P<bday>{_DAY})(?:,?\s+(?P<byear>{_YEAR}))?\b", re.I)
_WEEK_OF_RE = re.compile(rf"\bweek\s+of\s+(?:the\s+)?{_date_re('a')}", re.I)
_ONE_DATE_RE = re.compile(rf"\b{_date_re('a')}(?![0-9/])", re.I)
_REL_DAY_RE = re.compile(r"\b(?P<w>(?:the\s+)?day\s+before\s+yesterday|yesterday|last\s+night|today|tonight|"
                         r"this\s+(?:morning|afternoon|evening))\b", re.I)
_ROLLING_RE = re.compile(rf"\b(?:the\s+)?(?:last|past)\s+(?:(?P<n>{_NUM})\s+)?(?P<unit>{_UNIT}){_HEDGE_AFTER}\b",
                         re.I)
_AGO_RE = re.compile(rf"\b(?P<n>{_NUM})\s+(?P<unit>{_UNIT})(?P<hedge>{_HEDGE_AFTER})\s+ago\b", re.I)
_WEEKEND_RE = re.compile(r"\b(?P<which>last|this|past)\s+weekend\b", re.I)
_WEEKDAY_RE = re.compile(r"\b(?P<which>last|on|this\s+past|past)\s+(?P<wd>" + "|".join(_WEEKDAYS) + r")\b", re.I)
_MONTH_RE = re.compile(
    rf"\b(?P<prep>in|during|since|from|through|until|till|of|early|mid|late|end\s+of|start\s+of|beginning\s+of)"
    rf"\s+(?:the\s+)?(?P<part>early|mid|middle\s+of|late|end\s+of|start\s+of|beginning\s+of)?[\s-]*"
    rf"(?P<mon>{_MON_FULL})(?:,?\s+(?P<year>{_YEAR}))?\b", re.I)
_YEAR_RE = re.compile(rf"\b(?P<prep>in|during|since)\s+(?P<year>{_YEAR})\b", re.I)
_PAST_STATE_RE = re.compile(r"\b(?:was|were|had|used\s+to)\b", re.I)
_SINCE_BEFORE = re.compile(r"\bsince\s+(?:the\s+)?$", re.I)


@dataclass(frozen=True)
class TimeWindow:
    """``[start, end)`` in UTC. ``kind`` names the grammar shape that matched (or ``model``)."""

    start: datetime
    end: datetime
    kind: str
    source: str = SOURCE_PARSER
    past_state: bool = False

    def contains(self, at: datetime | None, *, pad: timedelta = WINDOW_PAD) -> bool:
        if at is None:
            return False
        at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
        return self.start - pad <= at < self.end + pad

    def with_past_state(self, value: bool) -> "TimeWindow":
        return replace(self, past_state=bool(value))

    def trace(self, tz: str = DEFAULT_TZ) -> Dict[str, Any]:
        """For the recall trace: local dates and the shape, never the message's words."""
        z = local_zone(tz)
        last = (self.end - timedelta(microseconds=1)).astimezone(z).date()
        return {"start": self.start.astimezone(z).date().isoformat(), "end": last.isoformat(),
                "kind": self.kind, "source": self.source, "past_state": self.past_state}


def local_zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz or DEFAULT_TZ)
    except Exception:  # noqa: BLE001 -- a bad zone name falls back to Juniper's
        return ZoneInfo(DEFAULT_TZ)


def _num(text: str | None) -> float:
    if not text:
        return 1.0
    t = " ".join(text.lower().split()).replace("a couple of", "couple").replace("couple of", "couple")
    t = t.replace("a couple", "couple").replace("a few", "few")
    if t.isdigit():
        return float(t)
    return float(_NUMBERS.get(t, 1))


def _midnight(d: date, z: ZoneInfo) -> datetime:
    return datetime.combine(d, time(0), tzinfo=z)


def _days(d0: date, d1_inclusive: date, z: ZoneInfo, kind: str) -> TimeWindow:
    if d1_inclusive < d0:
        d0, d1_inclusive = d1_inclusive, d0
    return TimeWindow(_midnight(d0, z).astimezone(timezone.utc),
                      _midnight(d1_inclusive + timedelta(days=1), z).astimezone(timezone.utc), kind)


def _span(start: datetime, end: datetime, kind: str) -> TimeWindow:
    return TimeWindow(start.astimezone(timezone.utc), end.astimezone(timezone.utc), kind)


def _year_for(month: int, day: int, today: date) -> int:
    """A month-day with no year is the most recent one: this year, unless that is well ahead."""
    try:
        cand = date(today.year, month, day)
    except ValueError:
        cand = date(today.year, month, min(day, 28))
    return today.year if cand <= today + timedelta(days=31) else today.year - 1


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _date_from(m: re.Match, prefix: str, today: date) -> date | None:
    g = m.groupdict()
    if g.get(prefix + "iso"):
        y, mo, d = (int(x) for x in g[prefix + "iso"].split("-"))
        return _safe_date(y, mo, d)
    if g.get(prefix + "us"):
        parts = g[prefix + "us"].split("/")
        mo, d = int(parts[0]), int(parts[1])
        y = int(parts[2]) if len(parts) > 2 else _year_for(mo, d, today)
        y = y + 2000 if y < 100 else y
        return _safe_date(y, mo, d)
    mon = g.get(prefix + "mon") or g.get(prefix + "mon2")
    day = g.get(prefix + "day") or g.get(prefix + "day2")
    year = g.get(prefix + "year") or g.get(prefix + "year2")
    if not mon or not day:
        return None
    mo = _MONTHS[mon.lower().rstrip(".")]
    d = int(re.sub(r"\D", "", day))
    return _safe_date(int(year) if year else _year_for(mo, d, today), mo, d)


def _since(text: str, m: re.Match, w: TimeWindow, now: datetime) -> TimeWindow:
    """"since <X>" runs from the start of X to now."""
    if _SINCE_BEFORE.search(text[: m.start()]) or m.group(0).lower().startswith("since"):
        return TimeWindow(w.start, now.astimezone(timezone.utc), "since_" + w.kind)
    return w


def _calendar_start(unit: str, today: date, back: int) -> date:
    if unit == "day":
        return today - timedelta(days=back)
    if unit == "week":
        return today - timedelta(days=today.weekday() + 7 * back)
    if unit == "month":
        y, mo = today.year, today.month - back
        while mo < 1:
            y, mo = y - 1, mo + 12
        return date(y, mo, 1)
    return date(today.year - back, 1, 1)


def parse_time_window(text: str, now: datetime, tz: str = DEFAULT_TZ) -> Optional[TimeWindow]:
    """The first time expression in ``text`` (in the shape order above) as a window, or None."""
    text = " ".join(str(text or "").split())
    if not text:
        return None
    z = local_zone(tz)
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(z)
    today = local_now.date()
    w = _parse(text, local_now, today, z)
    if w is None:
        return None
    ends_before_today = w.end <= _midnight(today, z).astimezone(timezone.utc)
    return w.with_past_state(ends_before_today and bool(_PAST_STATE_RE.search(text)))


def _parse(text: str, now: datetime, today: date, z: ZoneInfo) -> Optional[TimeWindow]:
    m = _RANGE_RE.search(text)
    if m:
        a = _date_from(m, "a", today)
        if a is not None:
            bmon = m.group("bmon")
            mo = _MONTHS[bmon.lower().rstrip(".")] if bmon else a.month
            by = int(m.group("byear")) if m.group("byear") else a.year
            b = _safe_date(by, mo, int(re.sub(r"\D", "", m.group("bday"))))
            if b is not None and b < a and not m.group("byear"):
                b = _safe_date(by + 1, mo, b.day) if bmon else None
            if b is not None and timedelta(0) <= b - a <= timedelta(days=366):
                return _days(a, b, z, "range")
    m = _WEEK_OF_RE.search(text)
    if m:
        a = _date_from(m, "a", today)
        if a is not None:
            return _days(a, a + timedelta(days=6), z, "week_of")
    m = _ONE_DATE_RE.search(text)
    if m:
        d = _date_from(m, "a", today)
        if d is not None:
            return _since(text, m, _days(d, d, z, "date"), now)
    m = _REL_DAY_RE.search(text)
    if m:
        word = " ".join(m.group("w").lower().split())
        if "before" in word:
            return _days(today - timedelta(days=2), today - timedelta(days=2), z, "relative_day")
        if word in ("yesterday", "last night"):
            return _days(today - timedelta(days=1), today - timedelta(days=1), z, "relative_day")
        return _span(_midnight(today, z), now, "relative_day")
    m = _AGO_RE.search(text)
    if m:
        unit = m.group("unit").lower().rstrip("s")
        days = _num(m.group("n")) * _UNIT_DAYS[unit]
        hedged = bool(m.group("hedge")) or bool(_HEDGE_BEFORE.search(text[: m.start()]))
        half = max(1.0, _UNIT_DAYS[unit] * _num(m.group("n")) * (0.5 if hedged else 0.35))
        if unit == "day":
            half = 1.0 if not hedged else max(1.0, days * 0.5)
        centre = now - timedelta(days=days)
        return _span(centre - timedelta(days=half), min(now, centre + timedelta(days=half)), "ago")
    m = _ROLLING_RE.search(text)
    if m:
        unit = m.group("unit").lower().rstrip("s")
        has_the = m.group(0).lower().startswith("the ") or bool(re.search(r"\b(?:over|in|during|for|within)\s+$",
                                                                          text[: m.start()], re.I))
        if m.group("n") or has_the or m.group(0).lower().startswith("past"):
            days = _num(m.group("n")) * _UNIT_DAYS[unit]
            hedged = "or" in m.group(0).lower().split() or m.group(0).lower().endswith("ish") \
                or bool(_HEDGE_BEFORE.search(text[: m.start()]))
            return _span(now - timedelta(days=days * (1.5 if hedged else 1.0)), now, "rolling")
        # bare "last week" / "last month": since the start of the previous calendar unit
        start = _calendar_start(unit, today, 1)
        return _span(_midnight(start, z), now, "calendar")
    m = re.search(rf"\bthis\s+(?P<unit>week|month|year)\b", text, re.I)
    if m:
        start = _calendar_start(m.group("unit").lower(), today, 0)
        return _span(_midnight(start, z), now, "calendar")
    m = _WEEKEND_RE.search(text)
    if m:
        monday = today - timedelta(days=today.weekday())
        if m.group("which").lower() == "this":
            sat = monday + timedelta(days=5)
            if sat > today:  # "this weekend" before it comes: the one just passed is the past one
                sat -= timedelta(days=7)
        else:
            sat = monday - timedelta(days=2)
        return _days(sat, sat + timedelta(days=1), z, "weekend")
    m = _WEEKDAY_RE.search(text)
    if m:
        wd = _WEEKDAYS[m.group("wd").lower()]
        back = (today.weekday() - wd) % 7
        if back == 0 and m.group("which").lower() != "on":
            back = 7
        d = today - timedelta(days=back)
        return _days(d, d, z, "weekday")
    m = _MONTH_RE.search(text)
    if m:
        mo = _MONTHS[m.group("mon").lower()]
        if m.group("year"):
            y = int(m.group("year"))
        else:
            y = today.year if date(today.year, mo, 1) <= today else today.year - 1
        last = calendar.monthrange(y, mo)[1]
        part = " ".join(((m.group("part") or "") or (m.group("prep") or "")).lower().split())
        d0, d1 = 1, last
        if part == "early" or part in ("start of", "beginning of"):
            d0, d1 = 1, 10
        elif part in ("mid", "middle of"):
            d0, d1 = 11, 20
        elif part in ("late", "end of"):
            d0, d1 = 21, last
        w = _days(date(y, mo, d0), date(y, mo, d1), z, "month")
        if m.group("prep").lower() == "since":
            return TimeWindow(w.start, now.astimezone(timezone.utc), "since_month")
        return w
    m = _YEAR_RE.search(text)
    if m:
        y = int(m.group("year"))
        w = _days(date(y, 1, 1), date(y, 12, 31), z, "year")
        if m.group("prep").lower() == "since":
            return TimeWindow(w.start, now.astimezone(timezone.utc), "since_year")
        return w
    return None


def window_from_dates(start: date, end_inclusive: date, now: datetime, tz: str = DEFAULT_TZ,
                      *, past_state: bool = False) -> Optional[TimeWindow]:
    """A model-written window (``WHEN: <first day> to <last day>``), or None when it is not a
    plausible time the message named: reversed, in the future, too old or too wide."""
    z = local_zone(tz)
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    if end_inclusive < start:
        return None
    w = _days(start, end_inclusive, z, "model")
    if w.start > now + timedelta(days=1) or w.end < now - MAX_AGE or w.end - w.start > MODEL_MAX_SPAN:
        return None
    end = min(w.end, now.astimezone(timezone.utc) + timedelta(days=1))
    ends_before_today = end <= _midnight(now.astimezone(z).date(), z).astimezone(timezone.utc)
    return TimeWindow(w.start, end, "model", SOURCE_MODEL, past_state=bool(past_state) and ends_before_today)


__all__ = ["DEFAULT_TZ", "SOURCE_MODEL", "SOURCE_PARSER", "TimeWindow", "WINDOW_PAD", "parse_time_window",
           "window_from_dates"]
