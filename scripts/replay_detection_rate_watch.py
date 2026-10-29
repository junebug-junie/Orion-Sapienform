#!/usr/bin/env python3
"""Replay the blind-detector watch (orion/vision_detection_rate_watch.py) over
stored ``vision_scene_inventory`` history, one evaluation per cron tick.

Read-only. Fetches each stream's (observed_at, saw-something) rows once and
evaluates in memory with the exact ``evaluate_detection_rate`` the ladder
watch runs, so the numbers here are what the live watch would have said.

Usage:
    python scripts/replay_detection_rate_watch.py --start 2026-10-16 --end 2026-10-29
    python scripts/replay_detection_rate_watch.py --start 2026-10-16 --end 2026-10-29 --json
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from orion import vision_detection_rate_watch as dr  # noqa: E402


def _connect(dsn):
    import psycopg2

    if dsn:
        conn = psycopg2.connect(dsn, connect_timeout=10)
    else:
        conn = psycopg2.connect(
            host=os.environ.get("ORION_PG_HOST", "localhost"),
            port=int(os.environ.get("ORION_PG_PORT", "55432")),
            user=os.environ.get("ORION_PG_USER", "postgres"),
            password=os.environ.get("ORION_PG_PASSWORD", os.environ.get("PGPASSWORD", "postgres")),
            dbname=os.environ.get("ORION_PG_DB", "conjourney"),
            connect_timeout=10,
        )
    conn.set_session(readonly=True, autocommit=True)
    return conn


class Series:
    """Sorted (epoch seconds, saw-something) with prefix sums for O(log n) span counts."""

    def __init__(self, rows):
        self.ts = [float(t) for t, _ in rows]
        self.cum = [0]
        for _, det in rows:
            self.cum.append(self.cum[-1] + (1 if det else 0))

    def count(self, lo: float, hi: float) -> dr.SpanCount:
        # (lo, hi] like DETECTION_RATE_SQL
        i = bisect.bisect_right(self.ts, lo)
        j = bisect.bisect_right(self.ts, hi)
        return dr.SpanCount(j - i, self.cum[j] - self.cum[i])


def evaluate_at(series: Series, stream: str, end: datetime, cfg: dr.DetectionRateConfig) -> dr.DetectionRateResult:
    span = cfg.span.total_seconds()
    e = end.timestamp()
    current = series.count(e - span, e)
    baseline = [series.count(e - d * 86400 - span, e - d * 86400) for d in range(1, cfg.baseline_days + 1)]
    return dr.evaluate_detection_rate(stream, current, baseline, cfg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=os.environ.get("POSTGRES_URI"))
    ap.add_argument("--start", required=True, help="first day (UTC, YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="day after the last (UTC, YYYY-MM-DD)")
    ap.add_argument("--tick-min", type=int, default=10)
    ap.add_argument("--streams", default=",".join(dr.DETECTION_RATE_STREAMS))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratio", type=float, default=dr.DEFAULT_CONFIG.ratio, help="sensitivity sweep")
    ap.add_argument("--floor", type=float, default=dr.DEFAULT_CONFIG.floor, help="sensitivity sweep")
    ap.add_argument("--span-hours", type=float, default=dr.DEFAULT_CONFIG.span.total_seconds() / 3600)
    args = ap.parse_args(argv)

    cfg = dataclasses.replace(
        dr.DEFAULT_CONFIG, ratio=args.ratio, floor=args.floor, span=timedelta(hours=args.span_hours)
    )
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    fetch_from = start - timedelta(days=cfg.baseline_days) - cfg.span
    conn = _connect(args.dsn)
    out = {}
    try:
        for stream in [s.strip() for s in args.streams.split(",") if s.strip()]:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT extract(epoch FROM observed_at), counts <> '{}'::jsonb FROM vision_scene_inventory "
                    "WHERE stream_id = %s AND observed_at > %s AND observed_at <= %s ORDER BY observed_at",
                    [stream, fetch_from, end],
                )
                series = Series(cur.fetchall())
            per_day = defaultdict(lambda: defaultdict(int))
            red_hours = defaultdict(set)
            t = start
            while t < end:
                r = evaluate_at(series, stream, t, cfg)
                day = t.strftime("%Y-%m-%d")
                per_day[day][r.status] += 1
                if r.red:
                    red_hours[day].add(t.hour)
                t += timedelta(minutes=args.tick_min)
            out[stream] = {
                d: {**dict(c), "red_utc_hours": sorted(red_hours[d])} for d, c in sorted(per_day.items())
            }
    finally:
        conn.close()

    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"span={cfg.span} baseline_days={cfg.baseline_days} floor={cfg.floor} ratio={cfg.ratio} "
          f"min_current_windows={cfg.min_current_windows} tick={args.tick_min}m")
    for stream, days in out.items():
        print(f"== {stream}")
        for d, c in days.items():
            print(f"{d}  collapsed={c.get('collapsed', 0):3d}  ok={c.get('ok', 0):3d}  "
                  f"unknown={c.get('unknown', 0):3d}  red UTC hours={c['red_utc_hours']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
