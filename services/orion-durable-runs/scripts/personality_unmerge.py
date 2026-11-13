#!/usr/bin/env python3
"""Roll back ONE self-naming MERGE (B was merged into A by ``graph.apply_namings``). Dry run by default.

    python services/orion-durable-runs/scripts/personality_unmerge.py --disposition <B id>          # show the plan
    python services/orion-durable-runs/scripts/personality_unmerge.py --disposition <B id> --apply  # write it

DSN: ``--dsn`` or ``POSTGRES_URI``. Stop the personality loop first (``PERSONALITY_ENABLED=false`` and
restart durable-runs), or the next tick may race the write. The write is one transaction with
``SELECT ... FOR UPDATE`` on both rows.

What it does:
- B: back to its last status before the merge (from ``personality_weight_history``; ``candidate`` if
  none), ``retired_at``, ``merged_into``, ``orion_verdict``, ``naming_id`` and
  ``verdict_evidence_count`` cleared. B's own counters, centroid and ``lane_recent`` were never changed
  by the merge (``lane_recent`` was faded to the merge time, its ``updated_at``), so the next tick
  fades it on from there.
- A: B's ``lane_counts``, ``evidence_count``, ``chosen_count`` and ``centroid_n`` subtracted; B's
  ``lane_recent`` (faded from B's ``updated_at`` to A's) subtracted per key, floored at 0; centroid
  un-mixed (A * n_A - B * n_B, renormalised); weight re-derived.
- One history row for each at now.

APPROXIMATE for A, and the plan says so: the merge clamped ``lane_recent`` at the per-lane cap, so
the subtraction cannot recover what the clamp cut, and evidence folded into A after the merge stays
on A (B's evidence rows always kept ``disposition_id = B``). The naming log row is never touched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.personality.graph import DEFAULT_PARAMS, Node, _derived_weight, _unit  # noqa: E402

from app.personality_store import (  # noqa: E402
    INSERT_HISTORY_SQL,
    NODE_COLUMNS,
    UPSERT_NODE_SQL,
    node_from_row,
    node_row,
)


def unmerge_plan(a: Node, b: Node, b_prev_status: str, now: datetime) -> tuple[Node, Node, list[str]]:
    """Pure: (restored A, restored B, notes). Raises ValueError when B is not merged into A."""
    if b.status != "retired" or b.merged_into != a.disposition_id:
        raise ValueError(f"{b.disposition_id} is not merged into {a.disposition_id}")
    notes: list[str] = []
    params = DEFAULT_PARAMS
    na = replace(a, centroid=list(a.centroid), lane_counts=dict(a.lane_counts), lane_recent=dict(a.lane_recent))
    for lane, c in b.lane_counts.items():
        left = na.lane_counts.get(lane, 0) - c
        if left > 0:
            na.lane_counts[lane] = left
        else:
            na.lane_counts.pop(lane, None)
    na.evidence_count = max(0, a.evidence_count - b.evidence_count)
    na.chosen_count = max(0, a.chosen_count - b.chosen_count)
    dt = (a.updated_at - b.updated_at).total_seconds() / 86400.0 if a.updated_at and b.updated_at else 0.0
    factor = 0.5 ** (max(0.0, dt) / params.half_life_days) if params.half_life_days > 0 else 1.0
    cap = math.expm1(params.lane_cap)
    for k, v in b.lane_recent.items():
        if na.lane_recent.get(k, 0.0) >= cap - 1e-9:
            notes.append(f"lane_recent[{k}] was at the cap: the clamp cut part of the merge, A's value is a floor")
        left = na.lane_recent.get(k, 0.0) - v * factor
        if left > 1e-9:
            na.lane_recent[k] = left
        else:
            na.lane_recent.pop(k, None)
    na.weight = _derived_weight(na.lane_recent, params)
    n_left = a.centroid_n - b.centroid_n
    ua, ub = _unit(a.centroid), _unit(b.centroid)
    if n_left > 0 and ua is not None and ub is not None and len(ua) == len(ub):
        mixed = _unit([x * a.centroid_n - y * b.centroid_n for x, y in zip(ua, ub)])
        if mixed is not None:
            na.centroid, na.centroid_n = mixed, n_left
    else:
        notes.append("centroid left as is (not enough centroid_n to un-mix)")
    na.updated_at = now
    nb = replace(b, status=b_prev_status, retired_at=None, merged_into=None, orion_verdict=None, naming_id=None,
                 verdict_evidence_count=None)
    na.history_weight, nb.history_weight = na.weight, nb.weight
    notes.append("evidence folded into A after the merge stays on A")
    return na, nb, notes


def _summary(n: Node) -> dict:
    return {"disposition_id": n.disposition_id, "status": n.status, "weight": round(n.weight, 4),
            "evidence_count": n.evidence_count, "lane_counts": n.lane_counts,
            "lane_recent": {k: round(v, 4) for k, v in n.lane_recent.items()}, "merged_into": n.merged_into,
            "orion_verdict": n.orion_verdict, "centroid_n": n.centroid_n}


def run(dsn: str, b_id: str, apply: bool, now: datetime) -> dict:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.transaction():
            lock = " FOR UPDATE" if apply else ""
            b_row = conn.execute(f"SELECT {NODE_COLUMNS} FROM personality_disposition WHERE disposition_id = %s{lock}",
                                 (b_id,)).fetchone()
            if b_row is None or not b_row.get("merged_into"):
                raise SystemExit(f"{b_id}: no such disposition, or it was not merged")
            a_row = conn.execute(f"SELECT {NODE_COLUMNS} FROM personality_disposition WHERE disposition_id = %s{lock}",
                                 (b_row["merged_into"],)).fetchone()
            if a_row is None:
                raise SystemExit(f"merge target {b_row['merged_into']} not found")
            prev = conn.execute("SELECT status FROM personality_weight_history WHERE disposition_id = %s AND at < %s "
                                "ORDER BY at DESC LIMIT 1", (b_id, b_row["retired_at"])).fetchone()
            a, b = node_from_row(a_row), node_from_row(b_row)
            na, nb, notes = unmerge_plan(a, b, (prev or {}).get("status") or "candidate", now)
            if na.status == "retired":
                notes.append("A is retired now; it stays retired")
            plan = {"apply": apply, "before": {"a": _summary(a), "b": _summary(b)},
                    "after": {"a": _summary(na), "b": _summary(nb)}, "notes": notes}
            if apply:
                for n in (na, nb):
                    conn.execute(UPSERT_NODE_SQL, node_row(n))
                    conn.execute(INSERT_HISTORY_SQL, {"disposition_id": n.disposition_id, "at": now,
                                                      "weight": float(n.weight), "status": n.status,
                                                      "evidence_count": int(n.evidence_count)})
    return plan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--disposition", required=True, help="the merged-away node (B)")
    ap.add_argument("--dsn", default=os.getenv("POSTGRES_URI", ""))
    ap.add_argument("--apply", action="store_true", help="write the plan (default: dry run)")
    args = ap.parse_args(argv)
    if not args.dsn:
        ap.error("--dsn or POSTGRES_URI is required")
    plan = run(args.dsn, args.disposition, args.apply, datetime.now(timezone.utc))
    print(json.dumps(plan, indent=2, default=str))
    if not args.apply:
        print("DRY RUN: nothing written. Re-run with --apply to write.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
