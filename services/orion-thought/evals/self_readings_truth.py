"""Truth-to-rows checker for the self-readings block (spec 2026-10-07 acceptance check 4).

Independent of ``orion.attention.self_readings``: it parses the RENDERED text, finds the row each
``[pe:<name>@<observed_at>]`` names among the persisted rows, and re-derives every number shown
from those rows with its own arithmetic. A number that does not resolve to a row, or does not
match it, is a failure. Used by the fixture eval (test_self_readings_truth_eval.py) and the live
replay (run_self_readings_live_eval.py).
"""

from __future__ import annotations

import bisect
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Sequence

Row = tuple[str, datetime, float]
WEEK = timedelta(days=7)
HOUR = timedelta(hours=1)
PREFIX = "node:substrate."

_REF = r"\[pe:(?P<name>[a-z_]+)@(?P<ts>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\]"
_TREND = r"(?:, (?:rising|settling))?"
LEVEL_RE = re.compile(
    r"^- (?P<label>.+?): (?P<score>\d+)/10 for me \(now (?P<value>\d+\.\d\d); higher than "
    rf"(?P<pct>\d+)% of my last 7 days{_TREND}\) {_REF}$"
)
RATE_RE = re.compile(
    r"^- (?P<label>.+?): (?P<score>\d+)/10 for me \(went off (?P<n>\d+) times? in the last hour; "
    rf"more than in (?P<pct>\d+)% of my hours this week{_TREND}\) {_REF}$"
)
TAIL_OK = re.compile(
    r"^(?:\d+ more unusual signals? not shown\. )?Nothing else I can read is unusual right now\."
    r"(?: No recent events: [a-z_]+(?:, [a-z_]+)*\.)?"
    r"(?: Can't read right now: [a-z_]+ \((?:no reading|stale|too little history)\)"
    r"(?:, [a-z_]+ \((?:no reading|stale|too little history)\))*\.)?$"
)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _score(pct: float) -> int:
    return int(min(10, max(0, math.floor(pct * 10.0 + 0.5))))


def _node_rows(rows: Sequence[Row], node: str, now: datetime) -> list[tuple[datetime, float]]:
    return sorted((_aware(t), float(v)) for n, t, v in rows
                  if n == node and _aware(t) <= now + timedelta(seconds=5) and math.isfinite(float(v)))


def _resolve(rows_n, ts: str, now: datetime) -> tuple[datetime, float] | None:
    """The newest row at or before now, which must carry the ref's timestamp (second precision)."""
    if not rows_n:
        return None
    newest = rows_n[-1]
    if newest[0].astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") != ts:
        return None
    return newest


def _rate_percentile(fires: list[datetime], value: int, now: datetime) -> float:
    horizon = now - WEEK
    grid = []
    t = horizon
    while t <= now:
        lo = bisect.bisect_right(fires, t - HOUR)
        hi = bisect.bisect_right(fires, t)
        grid.append(hi - lo)
        t += timedelta(minutes=1)
    return sum(1 for c in grid if c < value) / len(grid)


def check_block(text: str, rows: Sequence[Row], now: datetime) -> list[str]:
    """Failure lines (empty = every number in ``text`` resolves to ``rows`` and matches)."""
    now = _aware(now)
    failures: list[str] = []
    if not text:
        return failures
    lines = text.splitlines()
    for line in lines:
        if TAIL_OK.match(line):
            continue
        m = LEVEL_RE.match(line)
        kind = "level"
        if m is None:
            m = RATE_RE.match(line)
            kind = "rate"
        if m is None:
            failures.append(f"unparseable line (a number with no row behind it?): {line!r}")
            continue
        node = PREFIX + m["name"]
        rows_n = _node_rows(rows, node, now)
        if kind == "rate":
            # A trigger's ref names its newest ON row (the count is about the ON transitions).
            row = _resolve([r for r in rows_n if r[1] > 0.0], m["ts"], now)
        else:
            row = _resolve(rows_n, m["ts"], now)
        if row is None:
            failures.append(f"{node}: ref {m['ts']} is not its newest persisted row")
            continue
        week = [v for t, v in rows_n if t >= now - WEEK]
        if kind == "level":
            shown = float(m["value"])
            if abs(round(row[1], 2) - shown) > 0.0051:
                failures.append(f"{node}: shows {shown}, row holds {row[1]}")
            pct = sum(1 for v in week if v < row[1]) / len(week)
        else:
            fires, prev = [], False
            for t, v in rows_n:  # off->on transitions, re-derived here, not imported
                if v > 0.0 and not prev:
                    fires.append(t)
                prev = v > 0.0
            n = sum(1 for t in fires if now - HOUR < t <= now)
            if n != int(m["n"]):
                failures.append(f"{node}: shows {m['n']} onsets, rows hold {n} in the last hour")
            pct = _rate_percentile(fires, int(m["n"]), now)
        if int(math.floor(pct * 100.0)) != int(m["pct"]):
            failures.append(f"{node}: shows {m['pct']}%, rows give {pct * 100:.2f}%")
        if _score(pct) != int(m["score"]):
            failures.append(f"{node}: shows {m['score']}/10, rows give {_score(pct)}/10")
    return failures
