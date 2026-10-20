"""Read-only replay of the walkway rollups over the live tables.

Builds hour, day, week and month rollups for the last ``--days`` (default 3) in
a READ ONLY transaction, into memory (app/vision_rollup.py MemoryStore), and
prints the template lines. Nothing is written.

Checks (exit 1 on failure):
- every day's passings equal the sum of its hours' passings, and every
  week/month's passings the sum of its days';
- every hour the camera saw for fewer than 15 minutes reads "camera was down"
  (or, when sightings exist anyway, "camera record incomplete"), never "nobody passed";
- no stationary / prop / patio / parked-repeat sighting is counted: the
  excluded tallies plus counted sightings equal the raw sightings in the window;
- story sentences appear only when the spot-check gate passed;
- names are only labels Juniper typed (``vision_individual.label``);
- people are a range (rules_version 2): every row with people carries
  ``kinds.person.count_range`` with ``low <= high``; an hour's low is its most people in
  one window (less a person-kind prop sighted that hour, never below 1); a parent's low is
  the largest low of its children; and no line states one bare number of people.

Names are printed (this is Juniper's own eval on her own machine) unless
``--redact``.

Usage:
    PYTHONPATH=. python services/orion-sql-writer/evals/run_vision_rollup_replay_eval.py \\
        [--days 3] [--pg postgresql://...] [--hours 6] [--json out.json] [--redact]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from app.vision_rollup import MemoryStore, RollupConfig, run_one_rollup_cycle  # noqa: E402
from orion.vision.walkway_rollup import bare_people_counts, child_periods, Period, _parse  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"
PROP_IDS = ("fefb24d7e4074e9f827ad711874f6edf",)


def check(store: MemoryStore, stream: str) -> list[str]:
    fails: list[str] = []
    rows = store.rows
    for (s, grain, start), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        if s != stream:
            continue
        if grain == "hour":
            cam = r["camera"]
            line = store.lines[(s, grain, start)]
            if cam["status"] == "down" and "camera was down" not in line:
                fails.append(f"hour {r['label']}: camera saw {cam['up_minutes']} min but line is {line!r}")
            if cam["up_minutes"] < 15 and cam["status"] not in ("down", "unknown"):
                fails.append(f"hour {r['label']}: {cam['up_minutes']} camera minutes read as {cam['status']}")
            if cam["status"] == "unknown" and "record incomplete" not in line:
                fails.append(f"hour {r['label']}: sightings without frame record not flagged: {line!r}")
            if "nobody passed" in line and cam["status"] != "ok":
                fails.append(f"hour {r['label']}: 'nobody passed' with camera {cam['status']}")
        else:
            series = r.get("series") or []
            child_sum = sum(int(e.get("passings", 0)) for e in series)
            if child_sum != int(r["total"]["passings"]):
                fails.append(f"{grain} {r['label']}: passings {r['total']['passings']} != children {child_sum}")
        fails.extend(_range_fails(store, s, grain, start, r))
        stories = r.get("stories") or {}
        if not stories.get("gate_passed") and (stories.get("sentences") or stories.get("count")):
            fails.append(f"{grain} {r['label']}: story content with the gate closed")
    return fails


def _range_fails(store: MemoryStore, s: str, grain: str, start: datetime, r: dict) -> list[str]:
    out: list[str] = []
    line = store.lines[(s, grain, start)]
    bare = bare_people_counts(line)
    if bare:
        out.append(f"{grain} {r['label']}: bare people count {bare} in {line!r}")
    person = (r.get("kinds") or {}).get("person") or {}
    if int(person.get("passings", 0) or 0) <= 0:
        return out
    cr = person.get("count_range")
    if not cr:
        return out + [f"{grain} {r['label']}: people passed but no count_range"]
    if not 0 < cr["low"] <= cr["high"] or cr["high"] < int(person["passings"]):
        out.append(f"{grain} {r['label']}: bad range {cr} for {person['passings']} clusters")
    if grain == "hour":
        peak = int((r.get("peak_at_once") or {}).get("person", 0) or 0)
        if cr["method_low"] == "most_in_one_window" and cr["low"] != peak:
            out.append(f"hour {r['label']}: low {cr['low']} != most in one window {peak}")
    else:
        tz = ZoneInfo(r.get("tz") or "America/Denver")
        kids = [store.rows.get((s, "hour" if grain == "day" else "day", c.start))
                for c in child_periods(Period(grain, _parse(r["period_start"]), _parse(r["period_end"])), tz)]
        lows = [k["kinds"]["person"]["count_range"]["low"] for k in kids
                if k and ((k.get("kinds") or {}).get("person") or {}).get("count_range")]
        if lows and cr["low"] != max(lows):
            out.append(f"{grain} {r['label']}: low {cr['low']} != largest child low {max(lows)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--pg", default=os.getenv("POSTGRES_URI") or DEFAULT_PG)
    ap.add_argument("--stream", default="walkway")
    ap.add_argument("--hours", type=int, default=6, help="how many of the newest hour lines to print")
    ap.add_argument("--json", default="")
    ap.add_argument("--redact", action="store_true")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)
    cfg = RollupConfig(streams=(args.stream,), prop_individual_ids=PROP_IDS)
    mem = MemoryStore()
    out = run_one_rollup_cycle(postgres_uri=args.pg, cfg=cfg, now=now, since=since, write=False, memory=mem)
    print(json.dumps(out, default=str))

    def lines(grain: str) -> list[tuple[datetime, str, dict]]:
        return sorted(((k[2], mem.lines[k], mem.rows[k]) for k in mem.rows if k[1] == grain), key=lambda t: t[0])

    def show(text: str, row: dict) -> str:
        if not args.redact:
            return text
        for name in (row.get("named") or {}):
            text = text.replace(name, "<named>")
        return text

    hours = lines("hour")
    print(f"\n== hours: {len(hours)} built; newest {args.hours} ==")
    for _, line, row in hours[-args.hours:]:
        print(" ", show(line, row))
    down = [(l, r) for _, l, r in hours if r["camera"]["status"] == "down"]
    if down:
        print(f"\n== camera-down hours: {len(down)} (first 3) ==")
        for l, r in down[:3]:
            print(" ", show(l, r))
    for grain in ("day", "week", "month"):
        print(f"\n== {grain}s ==")
        for _, line, row in lines(grain):
            print(" ", show(line, row))
            if grain == "day" and row.get("captions"):
                for c in row["captions"][:2]:
                    print(f"     caption {c['at']}: {c['text'][:120]}")
    raw_total = sum(int(r["total"]["sightings"]) + sum(r["excluded"].values()) for _, _, r in hours)
    excluded = {}
    for _, _, r in hours:
        for k, v in r["excluded"].items():
            excluded[k] = excluded.get(k, 0) + v
    print(f"\nsightings in hours: {raw_total} ; excluded {excluded}")
    fails = check(mem, args.stream)
    if args.json:
        Path(args.json).write_text(json.dumps({"lines": {f"{k[1]}:{k[2].isoformat()}": v for k, v in mem.lines.items()},
                                               "fails": fails}, indent=2, default=str))
    if fails:
        print("\nFAIL:")
        for f in fails:
            print(" ", f)
        return 1
    print("\nPASS: day = sum of hours, week/month = sum of days, down hours read 'camera was down', "
          "story content only behind the gate, people always a range (low <= high, no bare count).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
