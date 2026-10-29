"""Read-only replay of street episodes (Temporal Self ``street`` arcs) over the last N days.

Runs the REAL chronicle read (``app.temporal_self_sources.SourceReader``, street reads only) and the
real reducer (``fold`` + ``advance_clock`` per window, like the chronicle) against Postgres in
read-only transactions, and prints per local day: street events read, street arcs, events folded into
an earlier arc of the same visit, events the daily cap kept out, and stories written after the
chronicle's read lag (lost as late rows under the old ended_at read).

Before/after: run it once from a checkout of the old code with ``--repo <that checkout>`` and once
from this one. Writes nothing.

    PYTHONPATH=. python services/orion-durable-runs/evals/run_street_episode_replay.py [--days 7]
        [--pg postgresql://...] [--repo <checkout>] [--cap 12]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[2]
DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"
READ_LAG_SEC = 300.0  # ChronicleConfig.read_lag_sec

LATE_WRITTEN_SQL = """
SELECT (s.created_at AT TIME ZONE %(tz)s)::date::text AS day, count(*) AS n
FROM vision_motion_story s
WHERE s.published AND s.clean AND s.ended_at >= %(lo)s AND s.ended_at < %(hi)s
  AND s.kind = ANY(%(kinds)s) AND s.stream_id <> ALL(%(room)s)
  AND s.created_at > s.ended_at + make_interval(secs => %(lag)s)
GROUP BY 1
"""


async def replay(pg: str, days: int, cap: int | None) -> int:
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from app.temporal_self_sources import SourceReader
    from orion.temporal_self import ReducerConfig, advance_clock, drain_closed_days, fold, initial_state
    from orion.temporal_self.day import DEFAULT_TZ, day_id_for, day_window
    from orion.temporal_self.sources import ROOM_STREAMS, STREET_STORY_KINDS

    cfg = ReducerConfig() if cap is None else ReducerConfig(street_arcs_per_day=cap)
    now = datetime.now(timezone.utc) - timedelta(seconds=READ_LAG_SEC)
    today = date.fromisoformat(day_id_for(now, DEFAULT_TZ))
    lo = day_window((today - timedelta(days=days)).isoformat(), DEFAULT_TZ)[0]
    events_by_day: Counter = Counter()
    seen: set[str] = set()
    arcs: dict = {}
    async with AsyncConnectionPool(pg, min_size=1, max_size=2, open=False, kwargs={
            "autocommit": True, "row_factory": dict_row,
            "options": "-c default_transaction_read_only=on"}) as pool:
        reader = SourceReader(pool, DEFAULT_TZ)
        state = initial_state()
        t = lo
        while t < now:
            hi = min(now, t + timedelta(hours=1))
            _, events = await reader.read(t, hi, ticks=False, kinds=["street_event"])
            for e in events:
                if e.event_id not in seen:
                    seen.add(e.event_id)
                    events_by_day[e.day_id] += 1
            state = advance_clock(fold(state, (), events, cfg), hi, cfg)
            state, closed = drain_closed_days(state)
            for d in closed:
                arcs.update((a.arc_id, a) for a in d.arcs if a.kind == "street")
            t = hi
        arcs.update((a.arc_id, a) for a in state.arcs.values() if a.kind == "street")
        async with pool.connection() as conn:
            cur = await conn.execute(LATE_WRITTEN_SQL, {
                "tz": DEFAULT_TZ, "lo": lo, "hi": now, "kinds": sorted(STREET_STORY_KINDS),
                "room": sorted(ROOM_STREAMS), "lag": READ_LAG_SEC})
            late = {r["day"]: r["n"] for r in await cur.fetchall()}

    per_day: dict[str, list] = {}
    for a in arcs.values():
        per_day.setdefault(a.day_id, []).append(a)
    print(f"street episode replay: {lo.isoformat()} .. {now.isoformat()}  cap={cfg.street_arcs_per_day}  "
          f"visit_gap={getattr(cfg, 'street_visit_gap_sec', 'n/a')}")
    print(f"{'day':<12}{'events':>8}{'arcs':>6}{'merged':>8}{'capped':>8}{'late_written':>14}")
    totals = Counter()
    for day in sorted(set(per_day) | set(events_by_day) | set(late)):
        day_arcs = per_day.get(day, [])
        in_arcs = sum(len(a.evidence_refs) for a in day_arcs)
        merged = in_arcs - len(day_arcs)
        capped = max(0, events_by_day.get(day, 0) - in_arcs)
        row = Counter(events=events_by_day.get(day, 0), arcs=len(day_arcs), merged=merged, capped=capped,
                      late_written=late.get(day, 0))
        totals.update(row)
        print(f"{day:<12}{row['events']:>8}{row['arcs']:>6}{row['merged']:>8}{row['capped']:>8}{row['late_written']:>14}")
    print(f"{'total':<12}{totals['events']:>8}{totals['arcs']:>6}{totals['merged']:>8}{totals['capped']:>8}"
          f"{totals['late_written']:>14}")
    for a in sorted(arcs.values(), key=lambda a: a.began_at):
        if len(a.evidence_refs) > 1:
            print(f"  visit {a.began_at.isoformat(timespec='seconds')}: {a.subject_label}  [{', '.join(a.evidence_refs)}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--pg", default=os.getenv("POSTGRES_URI") or DEFAULT_PG)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="checkout whose code to replay")
    ap.add_argument("--cap", type=int, default=None, help="street_arcs_per_day (default: ReducerConfig's)")
    args = ap.parse_args(argv)
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "services" / "orion-durable-runs")]
    return asyncio.run(replay(args.pg, args.days, args.cap))


if __name__ == "__main__":
    raise SystemExit(main())
