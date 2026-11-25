"""Replay the Juniper-reachable forecast over real chat history, held-out hour by hour (read-only).

Each hour from day 8 of the history on is forecast from messages strictly before it, with the same
code the service runs (app/reachable.py), then graded with the same verdict. Prints the overall and
cold-hour verdicts and the calibration table as JSON.

Run in the orion-prediction image, POSTGRES_URI and network from orion-athena-prediction:
    python evals/run_reachable_replay_eval.py [--days 120] [--no-hub]
Exit 0 always: the verdict is a report, not a gate (cold is expected to have no skill yet).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3])]

from app import reachable as R  # noqa: E402
from app.store import PostgresStore  # noqa: E402


def _paired(pairs) -> dict:
    """Hub (row p) vs no Hub (q) on the same origins: mean log loss and the daily gap (hub - no hub)."""
    if not pairs:
        return {"n_hours": 0}
    a, b = {}, {}
    for r, q in pairs:
        d = r["origin"].astimezone(R.LOCAL_TZ).date()
        a.setdefault(d, []).append(R.log_loss(r["p"], r["happened"]))
        b.setdefault(d, []).append(R.log_loss(q, r["happened"]))
    days = sorted(a)
    return {"n_hours": len(pairs), "n_days": len(days), "n_hours_with_message": sum(1 for r, _ in pairs if r["happened"]),
            "log_loss": {"hub": sum(map(sum, a.values())) / len(pairs), "no_hub": sum(map(sum, b.values())) / len(pairs)},
            "hub_vs_no_hub": R._gap(np.array([np.mean(a[d]) for d in days]), np.array([np.mean(b[d]) for d in days]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=120.0)
    ap.add_argument("--no-hub", action="store_true", help="replay without the Hub attention inputs (before/after)")
    a = ap.parse_args()
    now = R.hour_floor(datetime.now(timezone.utc))
    store = PostgresStore(os.environ["POSTGRES_URI"])
    msgs = store.juniper_messages(now - timedelta(days=a.days) - R.LOOKBACK, now)
    typed = store.juniper_claude_typed(now - timedelta(days=a.days) - R.LOOKBACK, now)
    coverage = store.juniper_typing_coverage(now - timedelta(days=a.days) - R.LOOKBACK - timedelta(hours=6), now)
    hub = [] if a.no_hub else store.juniper_hub_minutes(now - timedelta(days=a.days) - R.LOOKBACK - R.HUB_BACK, now)
    hub_since = None if a.no_hub else store.juniper_hub_since()
    store.close()
    if not msgs:
        print(json.dumps({"error": "no messages"}))
        return
    first = R.hour_floor(min(msgs)) + timedelta(days=7)
    start = max(first, now - timedelta(days=a.days))
    rows, o = [], start
    while o + R.WINDOW <= now:
        f = R.forecast(msgs, o, typed, coverage, hub, hub_since)
        rows.append({**f, "happened": R.outcome(msgs, o)[0]})
        o += R.WINDOW
    v = R.verdict(rows)
    # The fair test of the typing features: only origins the typing log covered.
    cov = R.verdict([r for r in rows if r["typing_log_covered"]])
    v["typing_covered_only"] = {k: cov[k] for k in ("all", "cold")}
    # The fair test of the Hub inputs: the same origins with and without them, over the full local
    # days the sensor covered (from the first Denver midnight after its first known hour).
    if hub_since is not None:
        first = (hub_since + R.HUB_BACK).astimezone(R.LOCAL_TZ)
        day0 = (first.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).astimezone(timezone.utc)
        paired = [(r, R.forecast(msgs, r["origin"], typed, coverage)["p"]) for r in rows if r["origin"] >= day0]
        v["hub_paired"] = {"from": day0.isoformat(), **{name: _paired([(r, q) for r, q in paired if keep(r)])
                           for name, keep in (("all", lambda r: True),
                                              ("cold", lambda r: r["state_bucket"] >= R.COLD_BUCKET))}}
    v["replay"] = {"from": start.isoformat(), "to": now.isoformat(), "messages": len(msgs), "claude_typed_windows": len(typed),
                   "hub_minutes": len(hub), "hub_since": hub_since,
                   "origins_typing_uncovered": sum(1 for r in rows if not r["typing_log_covered"])}
    print(json.dumps(v, indent=1, default=str))


if __name__ == "__main__":
    main()
