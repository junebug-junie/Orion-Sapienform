#!/usr/bin/env python3
"""Replay: which curiosity findings would have posted an event want (curiosity event wants step 3).

Runs the producer's own pure planner (``scripts.curiosity_event_wants.plan_findings``) over every philosophical
finding with a novelty, each ranked against the 30 days before it, and counts: no_history (under 20 earlier
findings), below_band (salience < 0.9), would_post (in the band).

* fixture (default, CI): a planted source of 60 findings over 30 days with distinct novelties. Gates: the first
  20 read no_history, and of the 40 with history, the share in the band is near a tenth (the band does what it
  says on a continuous magnitude), never every one.
* ``--live``: the real ``curiosity_finding_register`` rows (read-only, ``docker exec orion-athena-sql-db psql``).
  Prints the counts and the in-band runs. No gate: the live source is days old.

    PYTHONPATH=. python services/orion-hub/evals/run_curiosity_event_wants_replay.py [--live]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "services" / "orion-hub"))

from scripts.curiosity_event_wants import SALIENCE_BAND, plan_findings  # noqa: E402

PSQL = ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-Atc"]
LIVE_SQL = ("select json_build_object('run_id', run_id, 'line', line, 'novelty', novelty, 'created_at', created_at, "
            "'register', register) from curiosity_finding_register order by created_at")


def replay(rows: list[dict]) -> dict:
    """Counts over every philosophical row with a novelty (lookback = everything)."""
    phil = [r for r in rows if r.get("register", "philosophical") == "philosophical" and r["novelty"] is not None]
    if not phil:
        return {"rows": len(rows), "philosophical": 0, "counts": {}, "in_band": []}
    now = max(r["created_at"] for r in phil)
    plan = plan_findings(phil, now=now, lookback=now - min(r["created_at"] for r in phil) + timedelta(seconds=1))
    kind = Counter("no_history" if p["salience"] is None else ("would_post" if p["in_band"] else "below_band")
                   for p in plan)
    return {"rows": len(rows), "philosophical": len(phil), "counts": dict(kind),
            "in_band": [(p["run_id"], round(p["novelty"], 4), round(p["salience"], 3)) for p in plan if p["in_band"]]}


def fixture_rows() -> list[dict]:
    rng = random.Random(7)
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    return [{"run_id": f"r{k}", "line": "investigate", "novelty": round(0.10 + rng.random() * 0.1, 6),
             "created_at": t0 + timedelta(hours=12 * k), "register": "philosophical"} for k in range(60)]


def live_rows() -> list[dict]:
    out = subprocess.run(PSQL + [LIVE_SQL], check=True, capture_output=True, text=True).stdout
    rows = [json.loads(x) for x in out.splitlines() if x.strip()]
    for r in rows:
        r["created_at"] = datetime.fromisoformat(r["created_at"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    if args.live:
        res = replay(live_rows())
        print(json.dumps(res, indent=1, default=str))
        return 0
    res = replay(fixture_rows())
    print(json.dumps(res, indent=1, default=str))
    c = res["counts"]
    with_history = c.get("below_band", 0) + c.get("would_post", 0)
    share = c.get("would_post", 0) / with_history if with_history else 0.0
    ok = c.get("no_history") == 20 and with_history == 40 and 0.0 < share <= 2 * (1 - SALIENCE_BAND)
    print(f"band share {share:.3f} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
