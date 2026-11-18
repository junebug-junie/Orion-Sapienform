"""Live replay (READ-ONLY): which recent durable runs would have written an act, per family.

Body economy Patch A5, producer 1 (app/run_acts.py). Reads durable_admission_runs,
durable_resource_events and gpu_pool_events over a window and applies the SAME family map and
open/close rules the runtime uses:

* opens  = runs in a want family with a recorded ``run.resource_granted`` (first grant = open);
* closes = opened runs with a terminal (completed -> 1, failed / cancelled -> 0);
* still open = opened runs with no terminal yet (closed later by the run or by the ledger expiry);
* skipped = family runs that never got a grant (no work, no act);
* the correlation join: of the closed acts, how many have a released/expired/aborted pool lease whose
  ``turn_correlation_id`` is the run's correlation_id (what the act-ledger GPU cost fill joins on),
  and the slot-hours it would charge.

Writes nothing. Exits non-zero only if a check fails: in each family with >= 5 closes, at least 80%
of closed acts must join a lease inside the ledger's window (a hold lease is the run's grant), and
the family map must yield only the four want families.

    python services/orion-durable-runs/evals/run_acts_live_replay.py \
        --dsn postgresql://postgres:<pw>@localhost:55432/conjourney [--days 7]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE.parents[1]), str(SERVICE)]

import psycopg  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from datetime import timedelta  # noqa: E402

from app.run_acts import run_act_family  # noqa: E402
from orion.grammar.act_ledger import GPU_COST_CLOSE_GRACE_SEC, GPU_COST_LOOKBACK_SEC  # noqa: E402

RUNS_SQL = """
SELECT r.run_id, r.request, r.terminal, r.created_at,
       (SELECT min(generated_at) FROM durable_resource_events e
         WHERE e.run_id = r.run_id AND e.event = 'run.resource_granted') AS first_grant,
       (SELECT max(generated_at) FROM durable_resource_events e
         WHERE e.run_id = r.run_id AND e.entry_id = r.run_id || ':terminal:' || r.terminal) AS ended_at
  FROM durable_admission_runs r
 WHERE r.created_at > now() - make_interval(days => %(days)s)
"""
# What the act-ledger cost fill (orion/grammar/act_ledger.py fill_close_cost) would see at close:
# leases of the run's correlation_id whose release/expiry/abort event falls in
# [first grant - GPU_COST_LOOKBACK_SEC, terminal + GPU_COST_CLOSE_GRACE_SEC] AND whose row was already
# inserted by sql-writer when the close lands (terminal + CLOSE_APPLY_LAG). A hold released after
# the terminal (Door-A outreach keeps the hold for Hub) is counted as missed, not joined.
CLOSE_APPLY_LAG_SEC = 5.0
LEASES_SQL = """
SELECT count(*) AS leases, sum(held_ms) / 3600000.0 AS slot_h,
       count(*) FILTER (WHERE missed) AS missed, sum(held_ms) FILTER (WHERE missed) / 3600000.0 AS missed_h
  FROM (
    SELECT lease_id, max(held_ms) AS held_ms,
           bool_and(generated_at > %(until)s OR created_at > %(applied)s) AS missed
      FROM gpu_pool_events
     WHERE turn_correlation_id = %(corr)s AND event IN ('released', 'expired', 'aborted')
       AND held_ms IS NOT NULL AND role IS NOT NULL AND generated_at >= %(since)s
     GROUP BY lease_id) l
"""


def replay(dsn: str, days: int) -> dict:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        conn.read_only = True
        rows = conn.execute(RUNS_SQL, {"days": days}).fetchall()
        per: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        excluded: dict[str, int] = defaultdict(int)
        opened_corr: dict[str, str] = {}
        for row in rows:
            req = row["request"] or {}
            state = {"run_id": row["run_id"], "workflow": req.get("workflow"), "brief": req.get("brief") or {},
                     "admission": req.get("admission") or {}, "correlation_id": req.get("correlation_id")}
            family = run_act_family(state)
            if family is None:
                excluded[f"{req.get('workflow')}" + (":urgent" if (req.get("brief") or {}).get("urgent") else "")] += 1
                continue
            f = per[family.family]
            f["actor"], f["verb"] = family.actor, family.verb
            f["runs"] += 1
            if row["first_grant"] is None:
                f["skipped_never_granted"] += 1
                continue
            f["opens"] += 1
            if row["terminal"] and row["ended_at"] is not None:
                # Only a CLOSED act gets its GPU cost filled; a running hold has not released yet.
                opened_corr[row["run_id"]] = (family.family, state["correlation_id"],
                                              row["first_grant"], row["ended_at"])
                f["closes"] += 1
                f["closes_1" if row["terminal"] == "completed" else "closes_0"] += 1
                f.setdefault("_secs", []).append((row["ended_at"] - row["first_grant"]).total_seconds())
            else:
                f["still_open"] += 1
        for fam, corr, granted_at, ended_at in opened_corr.values():
            f = per[fam]
            hit = conn.execute(LEASES_SQL, {
                "corr": corr, "since": granted_at - timedelta(seconds=GPU_COST_LOOKBACK_SEC),
                "until": ended_at + timedelta(seconds=GPU_COST_CLOSE_GRACE_SEC),
                "applied": ended_at + timedelta(seconds=CLOSE_APPLY_LAG_SEC)}).fetchone()
            joined = int(hit["leases"] or 0) - int(hit["missed"] or 0)
            f["joined" if joined > 0 else "unjoined"] += 1
            f["slot_h"] = f.get("slot_h", 0.0) + float(hit["slot_h"] or 0.0) - float(hit["missed_h"] or 0.0)
            if hit["missed"]:
                f["acts_missing_a_lease"] += 1
                f["missed_slot_h"] = f.get("missed_slot_h", 0.0) + float(hit["missed_h"] or 0.0)
    out = {}
    for fam, f in sorted(per.items()):
        secs = sorted(f.pop("_secs", []))
        out[fam] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in f.items()}
        out[fam]["acts_per_day"] = round(f["opens"] / days, 2)
        out[fam]["median_seconds"] = round(secs[len(secs) // 2], 1) if secs else None
    return {"days": days, "families": out, "excluded_runs": dict(sorted(excluded.items()))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dsn", default=os.getenv("ORION_RUN_ACTS_REPLAY_DSN"))
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    if not args.dsn:
        print("--dsn (or ORION_RUN_ACTS_REPLAY_DSN) is required; the replay only reads", file=sys.stderr)
        return 2
    report = replay(args.dsn, args.days)
    print(json.dumps(report, indent=2, default=str))
    # A broken join key shows as most closes unjoined. A few misses are the known Door-A gap (the
    # hold outlives the terminal for Hub's outreach), reported as acts_missing_a_lease, not failed.
    failures = [fam for fam, f in report["families"].items()
                if f.get("closes", 0) >= 5 and f.get("joined", 0) < 0.8 * f["closes"]]
    wants = {"curiosity_investigate", "self_inquiry", "self_sense_eval", "journal_compose"}
    if failures:
        print(f"FAIL: under 80% of closed acts join a GPU lease (join key broken?): {failures}", file=sys.stderr)
    if not set(report["families"]) <= wants:
        print(f"FAIL: unexpected families {set(report['families']) - wants}", file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
