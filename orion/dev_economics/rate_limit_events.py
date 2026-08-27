"""Observe the rate limit -- and read the reset time it tells you.

WHY THIS EXISTS
---------------
`orion/autonomy/quota_budget.py` denominates scarcity in dollars against an
allowance. That allowance does not exist: no dollar threshold separates
limited from not-limited (see
`docs/superpowers/specs/2026-08-27-quota-window-calibration-finding.md`).

The constraint announces itself, in first-party text, at the moment it binds:

    You've hit your session limit · resets 3:30am (UTC)
    You've hit your weekly limit · resets Aug 18, 3am (UTC)

No denominator, no calibration, no proxy, and no knob -- an allowance set to
$450 silently stops refusing anything while still looking like a budget, but
"limited until 3:30am" cannot be tuned.

IT DOES NOT GIVE UP ANTICIPATION AFTER ALL
------------------------------------------
The reactive-versus-predictive trade this was expected to make turns out not to
be required. The message carries the reset time, so this reports not just that
the pool is empty but exactly when it refills -- an authoritative answer rather
than a forecast fitted to spend.

DETECTION IS STRUCTURAL, NOT SUBSTRING -- AND THAT MATTERS HERE
---------------------------------------------------------------
An event is a line with `isApiErrorMessage == true` whose message text matches
a known limit phrasing. Substring-matching `rate_limit_error` across the corpus
does NOT work, and the failure is not hypothetical: the first version of this
module reported 12 events in the current 5h window, and every one was this very
session's own tool output discussing rate limits. A detector that matches the
investigation that produced it is measuring itself.

`isApiErrorMessage` also covers errors that are NOT limits -- 401 auth
failures, "Prompt is too long", disabled subscription access. Those are
explicitly not limit events and must not be counted as scarcity.

THREE STATES, AND THE THIRD IS NOT A FAILURE
--------------------------------------------
`clear` / `limited` / `unknown`. `unknown` means no transcript activity was
observable in the window at all, which is NOT "not limited" -- the same
distinction `quota_budget.WindowSpend.observed` draws for spend. What to DO on
`unknown` is deliberately left to the caller: for a spend budget the safe
direction was to refuse, but here `unknown` usually means nobody has used
Claude recently, which is when the shared pool is least contended.

THE TWO CLOCKS, MEASURED THE HARD WAY
-------------------------------------
This reads append-only files written by live sessions. A message's `timestamp`
is when it HAPPENED; the file's mtime is when it was last WRITTEN.

Investigating the ledger's 18 consecutive all-zero ticks on 2026-08-26
18:14-22:31 UTC, a check for "files modified during that window" returned zero
and was read as proof of genuine silence. It was not: those messages live in
long-lived session files still being appended today, so their mtime is *now*
and can never fall inside a past window. Reading message timestamps instead
found **192 messages carrying 5.3M tokens** in the window called silent.

So filtering candidates by `mtime >= window_start` is sound -- a file last
written before the window cannot contain a message inside it. The reverse
inference is false, and is the mistake above. Only the sound direction is used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Literal

from orion.dev_economics.claude_code_ingest import DEFAULT_PROJECTS_ROOT, _parse_timestamp

LimitState = Literal["clear", "limited", "unknown"]
EventKind = Literal["session_limit", "weekly_limit"]

# Cheap substring prefilters applied before any JSON parse. The corpus is
# ~1.2GB across ~1300 files, so parsing every line is not affordable.
_ERROR_MARKER = "isApiErrorMessage"
_TIMESTAMP_MARKER = '"timestamp"'

# Narrow sensor, not a cognition architecture (CLAUDE.md "no regex swamp"):
# pull the timestamp out without a full JSON parse on the hot path.
_TS_RE = re.compile(r'"timestamp"\s*:\s*"([^"]+)"')

# The limit phrasings, matched against the rendered message text only after the
# line is confirmed to be a real API error message.
_SESSION_LIMIT = "hit your session limit"
_WEEKLY_LIMIT = "hit your weekly limit"

# "resets 3:30am (UTC)" / "resets 8pm (UTC)" / "resets Aug 18, 3am (UTC)"
_RESET_RE = re.compile(
    r"resets\s+(?:(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2}),\s*)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<mer>am|pm)",
    re.IGNORECASE,
)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


@dataclass(frozen=True)
class RateLimitEvent:
    at: datetime
    kind: EventKind
    resets_at: datetime | None
    source: str


def _message_text(obj: dict) -> str:
    message = obj.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return " ".join(parts)
    return ""


def parse_reset_at(text: str, event_at: datetime) -> datetime | None:
    """Read the reset time the limit message states.

    A session limit names a time with no date ("resets 3:30am"), which means the
    next occurrence at or after the event -- rolling to tomorrow when the stated
    hour has already passed today. A weekly limit names a month and day.

    Returns None rather than guessing when the text carries no reset time.
    """
    match = _RESET_RE.search(text)
    if not match:
        return None

    hour = int(match.group("hour")) % 12
    if match.group("mer").lower() == "pm":
        hour += 12
    minute = int(match.group("minute") or 0)

    mon, day = match.group("mon"), match.group("day")
    if mon and day:
        month = _MONTHS.get(mon.lower())
        if month is None:
            return None
        candidate = datetime(event_at.year, month, int(day), hour, minute, tzinfo=timezone.utc)
        # A December event naming a January reset belongs to the next year.
        if candidate < event_at - timedelta(days=180):
            candidate = candidate.replace(year=event_at.year + 1)
        return candidate

    candidate = event_at.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate < event_at:
        candidate += timedelta(days=1)
    return candidate


def classify(obj: dict) -> tuple[EventKind, str] | None:
    """A limit event, or None. `isApiErrorMessage` alone is not enough.

    401 auth failures, "Prompt is too long" and disabled-subscription errors all
    carry the same flag and are not scarcity.
    """
    if obj.get("isApiErrorMessage") is not True:
        return None
    text = _message_text(obj)
    low = text.lower()
    if _WEEKLY_LIMIT in low:
        return "weekly_limit", text
    if _SESSION_LIMIT in low:
        return "session_limit", text
    return None


@dataclass(frozen=True)
class LimitObservation:
    """What the transcripts on disk say about the window that just passed."""

    window_hours: float
    window_start: datetime
    window_end: datetime

    events: tuple[RateLimitEvent, ...]
    latest_activity_at: datetime | None
    # Freshest message that was NOT itself an API error. A run of failed
    # retries must not read as recovery.
    latest_success_at: datetime | None
    observed_message_count: int
    scanned_file_count: int

    @property
    def latest_event(self) -> RateLimitEvent | None:
        return self.events[-1] if self.events else None

    @property
    def observed(self) -> bool:
        """False means UNOBSERVED, not quiet."""
        return self.observed_message_count > 0

    @property
    def state(self) -> LimitState:
        """`limited` only while the constraint is still in force.

        Two independent ways it lifts, and both are checked because either
        alone would latch:

        * the stated reset time has passed -- authoritative, first-party
        * real activity happened after the event -- observed recovery, which
          covers a limit whose message carried no parseable reset time
        """
        if not self.observed:
            return "unknown"
        event = self.latest_event
        if event is None:
            return "clear"
        if event.resets_at is not None:
            # Authoritative and sufficient on its own. The activity fallback
            # must NOT override it: activity after a limit event is often
            # RETRIES that also failed, so "something happened at 18:50" is no
            # evidence a limit stated to hold until 20:00 has lifted. Live at
            # 2026-08-14 19:00 the fallback did exactly that and reported
            # `clear` an hour inside a stated limit window.
            return "clear" if self.window_end >= event.resets_at else "limited"
        if self.latest_success_at is not None and self.latest_success_at > event.at:
            # Only reached when the message carried no parseable reset time.
            return "clear"
        return "limited"

    @property
    def resets_at(self) -> datetime | None:
        """When the pool refills, if a limit is currently in force and said so."""
        return self.latest_event.resets_at if self.state == "limited" and self.latest_event else None

    @property
    def seconds_until_reset(self) -> float | None:
        reset = self.resets_at
        if reset is None:
            return None
        return max(0.0, (reset - self.window_end).total_seconds())

    @property
    def event_count(self) -> int:
        """How often the limit bound in this window -- graded pressure.

        Distinct from `state`. Six events in five hours and one event five
        hours ago can both read `clear` now and mean very different things.
        """
        return len(self.events)

    @property
    def staleness_sec(self) -> float | None:
        """Age of the freshest observation. None when nothing was observed.

        Transcripts are flushed by live sessions, not synchronously, so a
        `clear` reading with large staleness is weaker evidence than a fresh
        one. Threshold on this rather than assuming `state` is instantaneous.
        """
        if self.latest_activity_at is None:
            return None
        return (self.window_end - self.latest_activity_at).total_seconds()


def candidate_files(
    window_start: datetime, root: Path | str = DEFAULT_PROJECTS_ROOT
) -> Iterator[Path]:
    """Files that could contain a message inside the window.

    Sound direction only: a file last written BEFORE the window cannot contain
    a message inside it. The reverse ("not written during the window, therefore
    nothing in it") is false for append-only session files -- see the module
    docstring.
    """
    root_path = Path(root)
    if not root_path.exists():
        return
    cutoff = window_start.timestamp()
    for path in sorted(root_path.rglob("*.jsonl")):
        try:
            if path.stat().st_mtime >= cutoff:
                yield path
        except OSError:
            continue


def scan_window(
    window_start: datetime, window_end: datetime, files: Iterable[Path]
) -> tuple[list[RateLimitEvent], datetime | None, datetime | None, int]:
    """(events, latest_activity_at, latest_success_at, observed_count)."""
    events: list[RateLimitEvent] = []
    latest_activity: datetime | None = None
    latest_success: datetime | None = None
    count = 0

    for path in files:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if _TIMESTAMP_MARKER not in line:
                continue
            # Timestamp via regex on the hot path; JSON is parsed only for the
            # handful of lines that could actually be an error message.
            # rfind + slice rather than a regex over the whole line: transcript
            # lines carrying tool output run to hundreds of KB, and scanning
            # them end to end was most of this function's cost.
            idx = line.rfind(_TIMESTAMP_MARKER)
            ts_match = _TS_RE.match(line, idx) if idx >= 0 else None
            if not ts_match:
                continue
            at = _parse_timestamp(ts_match.group(1))
            if at is None or not (window_start <= at <= window_end):
                continue
            count += 1
            if latest_activity is None or at > latest_activity:
                latest_activity = at
            is_error = _ERROR_MARKER in line
            if not is_error and (latest_success is None or at > latest_success):
                latest_success = at
            if not is_error:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            classified = classify(obj)
            if classified is None:
                continue
            kind, body = classified
            events.append(
                RateLimitEvent(
                    at=at, kind=kind, resets_at=parse_reset_at(body, at), source=path.name
                )
            )

    events.sort(key=lambda e: e.at)
    return events, latest_activity, latest_success, count


def observe(
    *,
    now: datetime | None = None,
    window_hours: float = 5.0,
    root: Path | str = DEFAULT_PROJECTS_ROOT,
) -> LimitObservation:
    """Read the transcripts on disk and report the window's limit state."""
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(hours=window_hours)
    files = list(candidate_files(start, root))
    events, latest_activity, latest_success, count = scan_window(start, end, files)
    return LimitObservation(
        window_hours=window_hours,
        window_start=start,
        window_end=end,
        events=tuple(events),
        latest_activity_at=latest_activity,
        latest_success_at=latest_success,
        observed_message_count=count,
        scanned_file_count=len(files),
    )


def was_rate_limited_recently(
    hours: float = 5.0,
    *,
    now: datetime | None = None,
    root: Path | str = DEFAULT_PROJECTS_ROOT,
) -> bool | None:
    """`True` limited, `False` clear, `None` unknown.

    `None` is a real answer, not an error. Collapsing it to `False` is how an
    unobservable window reads as an empty road.
    """
    state = observe(now=now, window_hours=hours, root=root).state
    if state == "unknown":
        return None
    return state == "limited"
