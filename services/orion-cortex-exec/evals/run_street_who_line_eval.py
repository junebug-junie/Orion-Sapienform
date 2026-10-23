#!/usr/bin/env python3
"""Eval: does the street "who is around" line put Juniper's names first?

Default (no DB): replays the rows behind the live line seen in
orion-athena-cortex-exec-chat on 2026-10-15 23:35 UTC,

    a vehicle I have seen on 2 days but have no name for, <name>, a vehicle I
    have seen on 2 days but have no name for, 4 unfamiliar bicycles, and 2 more

and checks the properties the fix sells.

--live: the same checks on the real walkway, read-only, with the in-window
named individuals read straight from the database so "every name is shown"
is checked against what is actually there.

Checks:
1. every name Juniper gave (within the cap) is shown, and before any unnamed
   phrase;
2. no phrase appears twice;
3. "and N more" only hides names when there are more names than the cap;
4. no bare number of people (spec PR #2763 decision 5): unnamed people are a
   range, "at least L, at most H", "at least N", "up to N" or "a person".

Usage (from repo root):
    python services/orion-cortex-exec/evals/run_street_who_line_eval.py
    SITUATION_PERCEPTION_DSN=postgresql+psycopg2://postgres:postgres@localhost:55432/conjourney \\
        python services/orion-cortex-exec/evals/run_street_who_line_eval.py --live

Exit 0 = pass, 1 = a check failed, 2 = the database could not be read.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from orion.situational import perception_reader as pr  # noqa: E402
from orion.vision.walkway_rollup import bare_people_counts  # noqa: E402

PREFIX = "On the walkway in the last 15 minutes: "
NOW = datetime(2026, 10, 15, 23, 35, tzinfo=timezone.utc)


def _row(iid, kind, label=None, days=1, minutes_ago=2):
    return {"individual_id": iid, "kind": kind, "label": label, "distinct_days": days,
            "last_at": NOW - timedelta(minutes=minutes_ago)}


FIXTURE = [
    _row("edcb5152", "vehicle", days=2, minutes_ago=1),
    _row("f5242d40", "person", label="Pip", minutes_ago=2),
    _row("9b1729bb", "vehicle", days=2, minutes_ago=3),
    *[_row(f"bike{i}", "bicycle", minutes_ago=4 + i) for i in range(4)],
    *[_row(f"p{i}", "person", minutes_ago=5 + i) for i in range(3)],
    *[_row(f"v{i}", "vehicle", minutes_ago=6 + i) for i in range(2)],
]


def check(line: str, names: list[str]) -> list[str]:
    failures: list[str] = []
    if not line.startswith(PREFIX):
        return [f"no who line: {line!r}"] if names else []
    failures += [f"bare people count: {b!r}" for b in bare_people_counts(line)]
    body = line[len(PREFIX):].rstrip(".")
    more = ""
    if ", and " in body and body.rsplit(", and ", 1)[1].endswith((" more", "others")):
        body, more = body.rsplit(", and ", 1)
    # "at least 1, at most 4 unfamiliar people" is one phrase, not two.
    parts = [p.strip() for p in re.split(r", (?!at most )", body)]
    if len(parts) != len(set(parts)):
        failures.append(f"a phrase repeats: {parts}")
    want = names[: pr._MAX_WHO_PHRASES]
    if parts[: len(want)] != want:
        failures.append(f"names not first: shown={parts} names={want}")
    if more and len(names) <= pr._MAX_WHO_PHRASES and any(n not in parts for n in names):
        failures.append("a name fell into 'and N more' while names fit the cap")
    return failures


def _live_names(now: datetime) -> list[str] | None:
    from sqlalchemy import text

    engine = pr._get_engine()
    if engine is None:
        return None
    stmt = text(
        "SELECT i.label FROM vision_individual_sighting s "
        "JOIN vision_individual i ON i.individual_id = s.individual_id "
        "WHERE s.stream_id = 'walkway' AND NOT s.zone_no_embed AND NOT s.stationary "
        "  AND coalesce(trim(i.label), '') <> '' "
        "  AND s.ended_at >= CAST(:cut AS timestamptz) "
        "  AND s.started_at <= CAST(:until AS timestamptz) "
        "GROUP BY i.label ORDER BY max(s.ended_at) DESC"
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(stmt, {
                "cut": now - timedelta(minutes=pr.STREET_RECENT_MINUTES),
                "until": now + timedelta(seconds=pr.STREET_CLOCK_GRACE_SECONDS),
            }).all()
    except Exception as exc:  # noqa: BLE001 -- unread, exit 2
        print("name read failed:", type(exc).__name__, exc)
        return None
    seen: list[str] = []
    for (label,) in rows:
        label = label.strip()
        if label.casefold() not in {s.casefold() for s in seen}:
            seen.append(label)
    return seen


def main(argv: list[str]) -> int:
    if "--live" in argv:
        now = datetime.now(timezone.utc)
        summary = pr.fetch_street_summary("walkway", now=now)
        if not summary.read_ok:
            print("UNREAD: the street could not be read")
            return 2
        line = next((x for x in summary.lines if x.startswith(PREFIX)), "")
        names = _live_names(now)
        if names is None:
            print("UNREAD: no engine")
            return 2
    else:
        line = pr.summarize_street(
            sightings=FIXTURE, expectations=[], unresolved=[], patio=None,
            now=NOW, tz=ZoneInfo("America/Denver"), people_windows=[{"counts": {"person": 2}}],
        )[0]
        names = ["Pip"]
    print(line or "(no who line)")
    print("names in window:", names)
    failures = check(line, names)
    for f in failures:
        print("FAIL:", f)
    print("PASS" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
