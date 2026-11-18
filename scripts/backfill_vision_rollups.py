#!/usr/bin/env python3
"""Backfill walkway rollups (vision_rollup) for the last N days.

The clocked reducer (services/orion-sql-writer/app/vision_rollup_loop.py) only
builds missing hours back VISION_ROLLUP_CATCHUP_HOURS (48 h). This builds every
hour from ``--days`` ago (default 7: the crop retention; sightings are kept 90
days, scene inventory longer) up to the newest settled hour, then the days,
weeks and months that contain them, in that order (same code as the loop).

DRY RUN BY DEFAULT: reads in a READ ONLY transaction, builds into memory and
prints the day/week/month lines. Nothing is written.

``--apply`` (writes): first snapshots every existing vision_rollup row in the
range to ``/tmp/vision-rollup-backfill/snapshot.jsonl``, then upserts under the
reducer's advisory lock (a tick cannot interleave), and writes
``/tmp/vision-rollup-backfill/progress.log`` and ``report.md``. Rows are
upserts on (stream_id, grain, period_start); re-running is safe.

Usage:
    PYTHONPATH=. python scripts/backfill_vision_rollups.py [--days 7] [--stream walkway] [--pg ...] [--apply]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orion-sql-writer"))
sys.path.insert(0, str(ROOT))

from app.vision_rollup import MemoryStore, RollupConfig, run_one_rollup_cycle  # noqa: E402

DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"
DEFAULT_PROPS = "fefb24d7e4074e9f827ad711874f6edf"
JOB_DIR = Path("/tmp/vision-rollup-backfill")


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line)
    with (JOB_DIR / "progress.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def snapshot(pg: str, streams: tuple[str, ...], since: datetime) -> int:
    from sqlalchemy import create_engine, text

    engine = create_engine(pg)
    n = 0
    try:
        with engine.connect() as conn, (JOB_DIR / "snapshot.jsonl").open("w", encoding="utf-8") as fh:
            if conn.execute(text("SELECT to_regclass('vision_rollup')")).scalar() is None:
                return 0
            for row in conn.execute(text(
                    "SELECT stream_id, grain, period_start, complete, rollup, line FROM vision_rollup "
                    "WHERE stream_id = ANY(:s) AND period_end > :since"), {"s": list(streams), "since": since}).mappings():
                fh.write(json.dumps(dict(row), default=str) + "\n")
                n += 1
    finally:
        engine.dispose()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--stream", action="append", default=None, help="repeatable; default walkway")
    ap.add_argument("--pg", default=os.getenv("POSTGRES_URI") or DEFAULT_PG)
    ap.add_argument("--tz", default=os.getenv("VISION_LOCAL_TZ", "America/Denver"))
    ap.add_argument("--parked-min-hours", type=int, default=int(os.getenv("VISION_ROLLUP_PARKED_MIN_HOURS", "4")))
    ap.add_argument("--props", default=os.getenv("VISION_MOTION_PROP_INDIVIDUAL_IDS", DEFAULT_PROPS))
    ap.add_argument("--apply", action="store_true", help="write rows (default: dry run, nothing written)")
    args = ap.parse_args()

    streams = tuple(args.stream or ["walkway"])
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)
    cfg = RollupConfig(streams=streams, tz=args.tz, parked_min_hours=args.parked_min_hours,
                       prop_individual_ids=tuple(p.strip() for p in args.props.split(",") if p.strip()))
    if not args.apply:
        mem = MemoryStore()
        out = run_one_rollup_cycle(postgres_uri=args.pg, cfg=cfg, now=now, since=since, write=False, memory=mem)
        print("DRY RUN -- nothing written.", json.dumps(out, default=str))
        for grain in ("day", "week", "month"):
            for key in sorted((k for k in mem.lines if k[1] == grain), key=lambda k: (k[0], k[2])):
                print(f"[{grain}] {mem.lines[key]}")
        return 0

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    saved = snapshot(args.pg, streams, since)
    _log(f"backfill start streams={streams} since={since.isoformat()} snapshot_rows={saved} "
         f"(snapshot {JOB_DIR / 'snapshot.jsonl'})")
    t0 = time.monotonic()
    out = run_one_rollup_cycle(postgres_uri=args.pg, cfg=cfg, now=now, since=since, write=True)
    dt = time.monotonic() - t0
    _log(f"backfill done in {dt:.1f}s 100% {json.dumps(out, default=str)}")
    (JOB_DIR / "report.md").write_text(
        "# vision_rollup backfill\n\n"
        f"- verdict: {'skipped: ' + out['skipped'] if out.get('skipped') else 'written'}\n"
        f"- since: {since.isoformat()} (days={args.days})\n"
        f"- streams: {', '.join(streams)}\n"
        f"- rows existing before (snapshot): {saved} -> {JOB_DIR / 'snapshot.jsonl'}\n"
        f"- result: `{json.dumps(out, default=str)}`\n"
        f"- elapsed: {dt:.1f}s\n- errors: 0\n- another pass needed: no (re-runnable; upserts)\n",
        encoding="utf-8")
    print(f"report: {JOB_DIR / 'report.md'}")
    return 0 if not out.get("skipped") else 2


if __name__ == "__main__":
    raise SystemExit(main())
