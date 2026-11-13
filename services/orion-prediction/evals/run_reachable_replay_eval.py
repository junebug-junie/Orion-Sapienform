"""Replay the Juniper-reachable forecast over real chat history, held-out hour by hour (read-only).

Each hour from day 8 of the history on is forecast from messages strictly before it, with the same
code the service runs (app/reachable.py), then graded with the same verdict. Prints the overall and
cold-hour verdicts and the calibration table as JSON.

Run in the orion-prediction image, POSTGRES_URI and network from orion-athena-prediction:
    python evals/run_reachable_replay_eval.py [--days 120]
Exit 0 always: the verdict is a report, not a gate (cold is expected to have no skill yet).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3])]

from app import reachable as R  # noqa: E402
from app.store import PostgresStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=120.0)
    a = ap.parse_args()
    now = R.hour_floor(datetime.now(timezone.utc))
    store = PostgresStore(os.environ["POSTGRES_URI"])
    msgs = store.juniper_messages(now - timedelta(days=a.days) - R.LOOKBACK, now)
    typed = store.juniper_claude_typed(now - timedelta(days=a.days) - R.LOOKBACK, now)
    coverage = store.juniper_typing_coverage(now - timedelta(days=a.days) - R.LOOKBACK - timedelta(hours=6), now)
    store.close()
    if not msgs:
        print(json.dumps({"error": "no messages"}))
        return
    first = R.hour_floor(min(msgs)) + timedelta(days=7)
    start = max(first, now - timedelta(days=a.days))
    rows, o = [], start
    while o + R.WINDOW <= now:
        f = R.forecast(msgs, o, typed, coverage)
        rows.append({**f, "happened": R.outcome(msgs, o)[0]})
        o += R.WINDOW
    v = R.verdict(rows)
    # The fair test of the typing features: only origins the typing log covered.
    cov = R.verdict([r for r in rows if r["typing_log_covered"]])
    v["typing_covered_only"] = {k: cov[k] for k in ("all", "cold")}
    v["replay"] = {"from": start.isoformat(), "to": now.isoformat(), "messages": len(msgs), "claude_typed_windows": len(typed),
                   "origins_typing_uncovered": sum(1 for r in rows if not r["typing_log_covered"])}
    print(json.dumps(v, indent=1, default=str))


if __name__ == "__main__":
    main()
