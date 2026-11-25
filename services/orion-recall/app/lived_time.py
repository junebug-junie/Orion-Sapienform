"""Time windows in a chat message, for lived-memory recall (issue #3674).

"What happened with Robin over the last month or so?" names a time. Recall used to ignore it:
older rows that fit the topic better filled the candidate pool, and the record from the named
month never reached the rerank (issue #3881). This module turns the time words in a message
into one window, ``[start, end)`` in UTC, resolved against "now" in Juniper's time zone
(``America/Denver`` by default).

**A small date grammar, not a word list.** A window is one of five shapes. When a message holds
more than one, the phrase that comes first wins:

1. a range of two dates ("November 2nd to 8th", "between Oct 3 and Oct 10", "the week of Nov 2");
2. one date ("October 30th", "Oct 18", "on 10/30", "2026-10-30", "the 14th of November"). A bare
   ``M/D`` needs a year or a word that introduces a date ("on", "since", ...): "8/10" and "1/2 cup"
   are a rating and a fraction;
3. a day relative to today ("yesterday", "the day before yesterday", "last night"; "this morning"
   only in the past tense). "today" and "tonight" are the present, never a past time to search;
4. a span relative to now:
   - ``the last/past N <unit>`` is a rolling span ending now ("over the last month or so");
   - ``N <unit> ago`` is a span centred on that point ("a month or so ago", "two weeks ago");
   - ``last/this <unit>`` starts at the beginning of the previous/current calendar unit and runs
     to now ("last week", "this month"), and ``last/this weekend`` is that Saturday and Sunday;
   - ``last <weekday>`` is the most recent such day ("on Friday" may be the coming one: no window);
5. a named month or year after a preposition ("in October", "early September", "since August",
   "in 2025"). A bare month name needs a preposition or a qualifier: "May I ask" is not May. A
   month still ahead this year is last year's only in the past tense ("where was I in
   December?"); otherwise it is a plan, and no window.

``N`` is a digit or a number word ("a", "a couple of", "a few", "two" .. "twelve"). A hedge
("or so", "about", "around", "roughly") widens a span by half. "since <X>" runs from X to now.

**Past state.** ``past_state`` is True when the window ends before today and the message asks
in the past tense with a state verb ("Where *was* I in early October?"). Recall then also reads
the as-of rows (``orion.memory.lived_memory.EPISODE_VALID_DURING_SQL``): facts true in the
window that an update has since retired.

**Why no model.** Asking the fast-lane query rewrite for a ``WHEN:`` line was measured on the 57
scoreboard questions (2026-11-25): every wording tried either stopped it writing searches (0-3
of 57, against 11 with the plain prompt) or was ignored (0 time lines). The window comes from
this grammar alone, so it never waits on, or times out with, the model.

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
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?![a-z])\.?")
# Full month names only (a bare month needs a preposition anyway).
_MON_FULL = r"(?:" + "|".join(k for k in _MONTHS if len(k) > 4 or k in ("may", "june", "july")) + r")(?![a-z])"
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


# "8/10", "3/4", "1/2 cup" are ratings and fractions: a bare M/D counts only with a year or after a
# word that introduces a date.
_US_LEAD = re.compile(r"\b(?:on|since|from|around|until|till|by|before|after|of)\s+(?:the\s+)?$", re.I)


_RANGE_RE = re.compile(
    rf"\b(?:between\s+|from\s+)?{_date_re('a')}\s*(?:to|through|thru|until|till|and|-|–|—)\s*"
    rf"(?:the\s+)?(?:(?P<bmon>{_MON})\s+)?(?P<bday>{_DAY})(?:,?\s+(?P<byear>{_YEAR}))?\b", re.I)
_WEEK_OF_RE = re.compile(rf"\bweek\s+of\s+(?:the\s+)?{_date_re('a')}", re.I)
_ONE_DATE_RE = re.compile(rf"\b{_date_re('a')}(?![0-9/])", re.I)
# "today" and "tonight" are nearly always the present ("how are you today?"), never a past time to
# search. "this morning" names one only in the past tense ("what did I say this morning?").
_REL_DAY_RE = re.compile(r"\b(?P<w>(?:the\s+)?day\s+before\s+yesterday|yesterday|last\s+night|"
                         r"this\s+(?:morning|afternoon|evening))\b", re.I)
_PAST_TENSE_RE = re.compile(r"\b(?:was|were|had|did|said|told|asked|mentioned|talked|went|saw|got|made|"
                            r"happened|used\s+to)\b", re.I)
_ROLLING_RE = re.compile(rf"\b(?:the\s+)?(?:last|past)\s+(?:(?P<n>{_NUM})\s+)?(?P<unit>{_UNIT}){_HEDGE_AFTER}\b",
                         re.I)
_AGO_RE = re.compile(rf"\b(?P<n>{_NUM})\s+(?P<unit>{_UNIT})(?P<hedge>{_HEDGE_AFTER})\s+ago\b", re.I)
_WEEKEND_RE = re.compile(r"\b(?P<which>last|this|past)\s+weekend\b", re.I)
# "on Friday" may be the coming one (a plan), so only "last"/"this past" name a past weekday.
_WEEKDAY_RE = re.compile(r"\b(?P<which>last|this\s+past|past)\s+(?P<wd>" + "|".join(_WEEKDAYS) + r")\b", re.I)
_MONTH_RE = re.compile(
    rf"\b(?P<prep>in|during|since|from|through|until|till|of|early|mid|late|end\s+of|start\s+of|beginning\s+of)"
    rf"\s+(?:the\s+)?(?P<part>early|mid|middle\s+of|late|end\s+of|start\s+of|beginning\s+of)?[\s-]*"
    rf"(?P<mon>{_MON_FULL})(?!\.?\s+{_DAY}\b)(?:,?\s+(?P<year>{_YEAR}))?\b", re.I)  # "October 30th" is a date
_YEAR_RE = re.compile(rf"\b(?P<prep>in|during|since)\s+(?P<year>{_YEAR})\b", re.I)
_PAST_STATE_RE = re.compile(r"\b(?:was|were|had|used\s+to)\b", re.I)
_SINCE_BEFORE = re.compile(r"\bsince\s+(?:the\s+)?$", re.I)


@dataclass(frozen=True)
class TimeWindow:
    """``[start, end)`` in UTC. ``kind`` names the grammar shape that matched."""

    start: datetime
    end: datetime
    kind: str
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
                "kind": self.kind, "past_state": self.past_state}


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
    return w.with_past_state(ends_before_today(w, now, tz) and bool(_PAST_STATE_RE.search(text)))


def _parse(text: str, now: datetime, today: date, z: ZoneInfo) -> Optional[TimeWindow]:
    """Every shape's first match; the phrase that comes first in the message wins (ties go to the
    more specific shape, in the order below)."""
    found = []
    for order, shape in enumerate(_SHAPES):
        got = shape(text, now, today, z)
        if got is not None:
            found.append((got[0], order, got[1]))
    if not found:
        return None
    found.sort(key=lambda x: (x[0], x[1]))
    return found[0][2]


Shape = Optional[tuple[int, TimeWindow]]


def _shape_range(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    for m in _RANGE_RE.finditer(text):
        a = _date_from(m, "a", today)
        if a is None or (m.group("aus") and not _us_ok(text, m)):
            continue
        bmon = m.group("bmon")
        mo = _MONTHS[bmon.lower().rstrip(".")] if bmon else a.month
        by = int(m.group("byear")) if m.group("byear") else a.year
        b = _safe_date(by, mo, int(re.sub(r"\D", "", m.group("bday"))))
        if b is not None and b < a and not m.group("byear"):
            b = _safe_date(by + 1, mo, b.day) if bmon else None
        if b is not None and timedelta(0) <= b - a <= timedelta(days=366):
            return m.start(), _days(a, b, z, "range")
    return None


def _us_ok(text: str, m: re.Match) -> bool:
    """A bare M/D is a date only with a year or after a lead word, and never a rating ("8/10")."""
    us = m.group("aus") or ""
    if us.count("/") == 2:
        return True
    if us.endswith("/10"):
        return False
    return bool(_US_LEAD.search(text[: m.start("aus")]))


def _shape_week_of(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    for m in _WEEK_OF_RE.finditer(text):
        a = _date_from(m, "a", today)
        if a is not None and not (m.group("aus") and not _us_ok(text, m)):
            return m.start(), _days(a, a + timedelta(days=6), z, "week_of")
    return None


def _shape_date(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    for m in _ONE_DATE_RE.finditer(text):
        if m.group("aus") and not _us_ok(text, m):
            continue
        d = _date_from(m, "a", today)
        if d is not None:
            return m.start(), _since(text, m, _days(d, d, z, "date"), now)
    return None


def _shape_relative_day(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    m = _REL_DAY_RE.search(text)
    if not m:
        return None
    word = " ".join(m.group("w").lower().split())
    if "before" in word:
        return m.start(), _days(today - timedelta(days=2), today - timedelta(days=2), z, "relative_day")
    if word in ("yesterday", "last night"):
        return m.start(), _days(today - timedelta(days=1), today - timedelta(days=1), z, "relative_day")
    if not _PAST_TENSE_RE.search(text):
        return None  # "this morning" in the present tense is now, not a past time
    return m.start(), _span(_midnight(today, z), now, "relative_day")


def _shape_ago(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    m = _AGO_RE.search(text)
    if not m:
        return None
    unit = m.group("unit").lower().rstrip("s")
    n = _num(m.group("n"))
    days = n * _UNIT_DAYS[unit]
    hedged = bool(m.group("hedge")) or bool(_HEDGE_BEFORE.search(text[: m.start()]))
    if unit == "day":
        half = 1.0 if not hedged else max(1.0, days * 0.5)
    else:
        half = max(1.0, days * (0.5 if hedged else 0.35))
    centre = now - timedelta(days=days)
    return m.start(), _span(centre - timedelta(days=half), min(now, centre + timedelta(days=half)), "ago")


def _shape_rolling(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    for m in _ROLLING_RE.finditer(text):
        unit = m.group("unit").lower().rstrip("s")
        led = bool(re.search(r"\b(?:over|in|during|for|within)\s+$", text[: m.start()], re.I))
        if unit == "day" and not m.group("n") and not led:
            continue  # "the last day of the trip" is not a span
        has_the = m.group(0).lower().startswith("the ") or led
        if m.group("n") or has_the or m.group(0).lower().startswith("past"):
            days = _num(m.group("n")) * _UNIT_DAYS[unit]
            hedged = "or" in m.group(0).lower().split() or m.group(0).lower().endswith("ish") \
                or bool(_HEDGE_BEFORE.search(text[: m.start()]))
            return m.start(), _span(now - timedelta(days=days * (1.5 if hedged else 1.0)), now, "rolling")
        # bare "last week" / "last month": since the start of the previous calendar unit
        return m.start(), _span(_midnight(_calendar_start(unit, today, 1), z), now, "calendar")
    return None


def _shape_this(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    m = re.search(r"\bthis\s+(?P<unit>week|month|year)\b", text, re.I)
    if not m:
        return None
    start = _calendar_start(m.group("unit").lower(), today, 0)
    return m.start(), _span(_midnight(start, z), now, "calendar")


def _shape_weekend(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    m = _WEEKEND_RE.search(text)
    if not m:
        return None
    monday = today - timedelta(days=today.weekday())
    if m.group("which").lower() == "this":
        sat = monday + timedelta(days=5)
        if sat > today:  # "this weekend" before it comes is a plan, not a past time
            return None
    else:
        sat = monday - timedelta(days=2)
    return m.start(), _days(sat, sat + timedelta(days=1), z, "weekend")


def _shape_weekday(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    m = _WEEKDAY_RE.search(text)
    if not m:
        return None
    back = (today.weekday() - _WEEKDAYS[m.group("wd").lower()]) % 7 or 7
    d = today - timedelta(days=back)
    return m.start(), _days(d, d, z, "weekday")


def _shape_month(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    m = _MONTH_RE.search(text)
    if not m:
        return None
    mo = _MONTHS[m.group("mon").lower()]
    if m.group("year"):
        y = int(m.group("year"))
    elif date(today.year, mo, 1) <= today:
        y = today.year
    elif _PAST_TENSE_RE.search(text):
        y = today.year - 1   # "where was I in December?" asked in November: last December
    else:
        return None          # "I'm seeing Robin in December": a plan, not a past time
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
        return m.start(), TimeWindow(w.start, now.astimezone(timezone.utc), "since_month")
    return m.start(), w


def _shape_year(text: str, now: datetime, today: date, z: ZoneInfo) -> Shape:
    m = _YEAR_RE.search(text)
    if not m:
        return None
    y = int(m.group("year"))
    if y > today.year:
        return None
    w = _days(date(y, 1, 1), date(y, 12, 31), z, "year")
    if m.group("prep").lower() == "since":
        return m.start(), TimeWindow(w.start, now.astimezone(timezone.utc), "since_year")
    return m.start(), w


# Most specific first: on a tie at the same position, the earlier shape wins.
_SHAPES = (_shape_range, _shape_week_of, _shape_date, _shape_relative_day, _shape_ago, _shape_rolling,
           _shape_this, _shape_weekend, _shape_weekday, _shape_month, _shape_year)


def ends_before_today(window: TimeWindow, now: datetime, tz: str = DEFAULT_TZ) -> bool:
    """The window is wholly past: it ends by local midnight today (the past-state rule)."""
    z = local_zone(tz)
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return window.end <= _midnight(now.astimezone(z).date(), z).astimezone(timezone.utc)


__all__ = ["DEFAULT_TZ", "TimeWindow", "WINDOW_PAD", "ends_before_today", "local_zone", "parse_time_window"]
