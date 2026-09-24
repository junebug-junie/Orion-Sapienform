#!/usr/bin/env python3
"""Read-only report on the walkway individuals + rhythm reducers.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md,
"Recommended next patch": answers "do appearance clusters hold across days"
from data inside the first week. Nothing here writes anything.

Prints:
  - distinct individuals seen per local day
  - sightings per individual (top N)
  - cluster count growth (new individuals per day, cumulative)
  - labeled count
  - open asks
  - expectation outcomes (met / missed / unscorable / open)

A cluster count that keeps growing roughly linearly with sightings means the
match threshold is too strict (every appearance is a "new" person); one that
never grows past a handful means it is too loose.

    python3 scripts/report_vision_individuals.py
    python3 scripts/report_vision_individuals.py --stream walkway --days 14 --json

DSN: --dsn, else POSTGRES_URI, else DATABASE_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


def build_report(conn, *, stream: str, days: int, tz: str, top: int) -> Dict[str, Any]:
    from sqlalchemy import text

    p = {"s": stream, "days": days, "tz": tz, "top": top}
    since = "now() - make_interval(days => :days)"
    per_day = conn.execute(text(f"""
        SELECT (s.started_at AT TIME ZONE :tz)::date AS day, count(DISTINCT s.individual_id), count(*)
        FROM vision_individual_sighting s WHERE s.stream_id = :s AND s.started_at > {since}
        GROUP BY 1 ORDER BY 1"""), p).fetchall()
    growth = conn.execute(text(f"""
        SELECT (first_seen_at AT TIME ZONE :tz)::date AS day, count(*)
        FROM vision_individual WHERE stream_id = :s AND first_seen_at > {since}
        GROUP BY 1 ORDER BY 1"""), p).fetchall()
    top_rows = conn.execute(text("""
        SELECT individual_id, kind, label, sighting_count, distinct_days, first_seen_at, last_seen_at
        FROM vision_individual WHERE stream_id = :s ORDER BY sighting_count DESC LIMIT :top"""), p).fetchall()
    totals = conn.execute(text("""
        SELECT count(*), count(label), COALESCE(sum(CASE WHEN sighting_count = 1 THEN 1 ELSE 0 END), 0)
        FROM vision_individual WHERE stream_id = :s"""), p).fetchone()
    asks = conn.execute(text("""
        SELECT ask_id, question, created_at, expires_at FROM orion_ask
        WHERE status = 'open' ORDER BY created_at"""), p).fetchall()
    outcomes = conn.execute(text(f"""
        SELECT status, count(*) FROM vision_percept_expectation
        WHERE stream_id = :s AND window_start > {since} GROUP BY status"""), p).fetchall()

    cumulative, run = [], 0
    for day, n in growth:
        run += n
        cumulative.append({"day": str(day), "new": n, "cumulative": run})
    return {
        "stream": stream,
        "window_days": days,
        "individuals_per_day": [{"day": str(d), "individuals": i, "sightings": s} for d, i, s in per_day],
        "cluster_growth": cumulative,
        "top_individuals": [
            {"individual_id": r[0], "kind": r[1], "label": r[2], "sightings": r[3], "distinct_days": r[4],
             "first_seen": r[5].isoformat(), "last_seen": r[6].isoformat()} for r in top_rows],
        "individuals_total": totals[0],
        "individuals_labeled": totals[1],
        "individuals_seen_once": totals[2],
        "open_asks": [{"ask_id": a[0], "question": a[1], "created_at": a[2].isoformat(),
                       "expires_at": a[3].isoformat() if a[3] else None} for a in asks],
        "expectation_outcomes": {s: n for s, n in outcomes},
    }


def _print(rep: Dict[str, Any]) -> None:
    print(f"Walkway individuals report -- stream={rep['stream']} last {rep['window_days']} days")
    print(f"\nIndividuals: {rep['individuals_total']} total, {rep['individuals_labeled']} named by Juniper, "
          f"{rep['individuals_seen_once']} seen only once")
    print("\nPer day (distinct individuals / sightings):")
    for r in rep["individuals_per_day"] or [{"day": "(none)", "individuals": 0, "sightings": 0}]:
        print(f"  {r['day']}  {r['individuals']:>4} / {r['sightings']}")
    print("\nCluster growth (new individuals per day, cumulative):")
    for r in rep["cluster_growth"] or [{"day": "(none)", "new": 0, "cumulative": 0}]:
        print(f"  {r['day']}  +{r['new']:<4} = {r['cumulative']}")
    print("\nMost-seen individuals:")
    for r in rep["top_individuals"]:
        name = r["label"] or f"{r['kind']} #{r['individual_id'][:6]}"
        print(f"  {name:<28} {r['sightings']:>4} sightings over {r['distinct_days']} days")
    print(f"\nOpen asks: {len(rep['open_asks'])}")
    for a in rep["open_asks"]:
        print(f"  {a['created_at'][:16]}  {a['question']}")
    o = rep["expectation_outcomes"]
    print("\nExpectations: " + ", ".join(f"{k}={o.get(k, 0)}" for k in ("met", "missed", "unscorable", "open")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL"))
    ap.add_argument("--stream", default="walkway")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--tz", default=os.getenv("VISION_LOCAL_TZ", "America/Denver"))
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not args.dsn:
        print("no DSN: pass --dsn or set POSTGRES_URI", file=sys.stderr)
        return 2
    from sqlalchemy import create_engine

    engine = create_engine(args.dsn)
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            rep = build_report(conn, stream=args.stream, days=args.days, tz=args.tz, top=args.top)
    finally:
        engine.dispose()
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        _print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
