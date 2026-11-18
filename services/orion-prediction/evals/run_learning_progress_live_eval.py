"""Learning progress on live scored forecasts, read-only (nothing is stored). Acceptance S1.3 and the metric gate's
live check (docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md, Step 1).

For every complete day since each series began, prints what ``GET /learning-progress`` would have said that day:
status, learning progress, its standard error, t and direction. The check: the signal reads ``flat`` on most days
(it can rest), it is never credited across a model-roster change, and it is not stuck (it can read better or
worse). The last line per series is today's reading, to compare with the endpoint and the hand SQL.

    cd services/orion-prediction && POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_learning_progress_live_eval.py

Baseline 2026-11-18: cabinet flat 8, insufficient 12 (model change 11-03), today -0.073 t -0.93 (cut 2.33) flat;
office better once (11-16, +0.228 t 2.86, cut 2.18), today +0.031 t 0.34 flat.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]

from app import learning_progress as lp  # noqa: E402
from app.store import PostgresStore  # noqa: E402


def main() -> int:
    store = PostgresStore(os.environ["POSTGRES_URI"])
    since = datetime(2026, 10, 1, tzinfo=timezone.utc)
    pairs = store.daily_skill_pairs(lp.SERIES, lp.HORIZON_MIN, since)
    rosters = store.daily_rosters(lp.SERIES, since)
    today = datetime.now(timezone.utc).date()
    bad = 0
    for sid in lp.SERIES:
        days = [lp.DayStat(r["day"], int(r["pairs"]), float(r["mae_champion"]), float(r["mae_baseline"]))
                for r in pairs if r["series_id"] == sid]
        roster_all = {r["day"]: frozenset(r["models"]) for r in rosters if r["series_id"] == sid}
        if not days:
            print(f"{sid}: no scored pairs")
            bad += 1
            continue
        first = min(d.day for d in days)
        print(f"\n{sid} (first scored day {first})")
        print("as_of       status        lp       se      t   t_crit  dir     recent  prior  roster_change")
        tally: Counter = Counter()
        d = first + timedelta(days=2 * lp.WINDOW_DAYS)
        while d <= today:
            # Only what was known that day: fits and scores before it.
            p = lp.progress(sid, [x for x in days if x.day < d], {k: v for k, v in roster_all.items() if k < d}, d)
            tally[p.direction or p.status] += 1
            f = (lambda v, w=7: f"{v:+.3f}".rjust(w) if v is not None else "-".rjust(w))
            print(f"{d}  {p.status:12} {f(p.learning_progress)} {f(p.se)} {f(p.t)} {f(p.t_crit, 6)}  {str(p.direction or '-'):7} "
                  f"{p.recent.days:>5}  {p.prior.days:>5}  {p.roster_changed_on or '-'}")
            d += timedelta(days=1)
        print(f"tally: {dict(tally)}")
        if tally.get("flat", 0) == 0 and tally.get("insufficient", 0) == 0:
            print("FAIL: never at rest")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
