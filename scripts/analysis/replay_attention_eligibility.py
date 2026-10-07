#!/usr/bin/env python3
"""Replay the workspace competition with candidate world signals added. Read-only.

Question (attend-to-act loop, item 2): if the cabinet warming signal (and, to judge honesty, the
RPC-delivery timeout pressure and circe's inference-failure pressure) entered the workspace broadcast
through the EXISTING mechanism -- a prediction-error node the dynamics engine turns into
``dynamic_pressure = 0.6 * prediction_error`` -- how often would each win, and what would it displace?

How: for every logged broadcast tick in the window (``substrate_attention_broadcast_log``), rebuild
the competing nodes from the logged open loops (label, evidence strength = node dynamic pressure,
evidence breadth via contributing ids), add the candidate node with its value AT THAT TIME, and run
the real pipeline (``substrate_pressure_signals`` -> ``merge_signals`` -> ``build_open_loops`` ->
``select_actions``). Top-down goal bias cannot be replayed (it reads the live goal): ticks whose
logged frame carried a voluntary override are reported separately and counted as "no change".
Fidelity check: the replay WITHOUT any candidate must reproduce the logged winner; the agreement
rate is printed so the reader can judge the rest.

    python scripts/analysis/replay_attention_eligibility.py --hours 72
    python scripts/analysis/replay_attention_eligibility.py --hours 72 --json

Needs ORION_PG_DSN (or POSTGRES_URI). Opens a read-only transaction with a statement timeout.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orion.autonomy.cabinet_heat import read_cabinet_heat  # noqa: E402
from orion.hardware_watch.rules import TempPoint  # noqa: E402
from orion.substrate.attention.policy import select_actions  # noqa: E402
from orion.substrate.attention.scoring import build_open_loops, merge_signals  # noqa: E402
from orion.substrate.attention_broadcast import substrate_pressure_signals  # noqa: E402

# orion/substrate/pressure.py: prediction_error seed = raw * 0.6 (decay ~1 for a node rewritten
# every 30 s).
PE_TO_DYNAMIC = 0.6
CANDIDATES = {
    "node:substrate.cabinet": "Cabinet prediction error",
    "node:substrate.rpc_delivery": "Rpc delivery prediction error",
    "node:circe": "Circe prediction error",
}


def classify(node: str | None) -> str:
    if not node:
        return "none"
    for key in ("chat", "biometrics", "execution", "harness_closure", "route", "codebase",
                "perception", "bus_synaptic", "cabinet", "rpc_delivery", "circe", "vision"):
        if key in node:
            return key
    return "other"


def _node(node_id: str, label: str, pressure: float, extra_refs: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id, label=label,
        metadata={"dynamic_pressure": pressure, "dynamic_pressure_reason": "prediction_error",
                  "contributing_turn_ids": [f"replay-{i}" for i in range(max(0, extra_refs))]},
        signals=SimpleNamespace(confidence=1.0))


def logged_nodes(frame: dict) -> list[SimpleNamespace]:
    out = []
    for loop in frame.get("open_loops") or []:
        refs = [r for r in loop.get("source_refs") or [] if str(r).startswith("node:")]
        if not refs:
            continue
        feats = loop.get("salience_features") or {}
        strength = float(feats.get("evidence_strength") or loop.get("salience") or 0.0)
        breadth = float(feats.get("evidence_breadth") or 0.5)
        # breadth = |distinct sources + refs| / 4; a bare node signal is 2 (source + node id)
        extra = max(0, round(breadth * 4) - 2)
        out.append(_node(refs[0], str(loop.get("description") or refs[0]), strength, extra))
    return out


def compete(nodes: list[SimpleNamespace], min_salience: float) -> tuple[str | None, str]:
    """Bottom-up winner (node id, action type) through the real pipeline, verdict lookup off."""
    signals = substrate_pressure_signals(nodes, min_salience=min_salience, limit=24)
    merged = merge_signals(signals, limit=15)
    loops = build_open_loops(signals=merged, ctx={}, inputs={}, belief_lineage=[], direct_turn=False,
                             stale_thread_active=False, max_open=5, verdict_lookup=None)
    _, selected, _, _ = select_actions(open_loops=loops, suppressions=[], min_ask=0.65, max_asks=0,
                                       stale_thread_active=False)
    if selected is None or not selected.open_loop_id:
        return None, "none"
    winner = next((lp for lp in loops if lp.id == selected.open_loop_id), None)
    node = next((r for r in (winner.source_refs if winner else []) if str(r).startswith("node:")), None)
    return node, str(getattr(selected, "action_type", "none"))


@dataclass
class Series:
    """A value over time, read as the newest sample at or before t (None past max_age_sec)."""

    ts: list[datetime] = field(default_factory=list)
    vals: list[float] = field(default_factory=list)
    max_age_sec: float = 300.0

    def at(self, t: datetime) -> float | None:
        i = bisect.bisect_right(self.ts, t) - 1
        if i < 0 or (t - self.ts[i]).total_seconds() > self.max_age_sec:
            return None
        return self.vals[i]


def cabinet_series(points: list[TempPoint], ticks: Iterable[datetime], threshold: float) -> dict[datetime, float]:
    ts = [p.ts for p in points]
    out = {}
    for t in ticks:
        lo = bisect.bisect_left(ts, t - timedelta(hours=2))
        hi = bisect.bisect_right(ts, t)
        out[t] = read_cabinet_heat(points[lo:hi], t, rise_threshold_c=threshold).warming_error
    return out


def replay(rows: list[dict], values: dict[str, Any], *, min_salience_default: float = 0.05) -> dict:
    """rows: broadcast log rows {generated_at: datetime, projection: dict}. values: node id ->
    callable(t) -> prediction_error in [0,1] or None. Pure; returns the report."""
    rep: dict[str, Any] = {"ticks": 0, "override_ticks": 0, "fidelity": {"agree": 0, "total": 0}, "candidates": {}}
    for node in values:
        rep["candidates"][node] = {"nonzero_ticks": 0, "admitted_ticks": 0, "wins": 0,
                                   "wins_with_action": 0, "bindable_ticks": 0, "displaced": Counter(), "longest_win_run": 0,
                                   "win_hours": Counter(), "max_value": 0.0}
    runs = {n: 0 for n in values}
    for row in rows:
        t, proj = row["generated_at"], row["projection"]
        frame = proj.get("frame") or {}
        rep["ticks"] += 1
        logged = (proj.get("attended_node_ids") or [None])[0]
        logged_action = str(proj.get("selected_action_type") or "none")
        min_sal = float((frame.get("debug") or {}).get("min_salience") or min_salience_default)
        base = logged_nodes(frame)
        override = bool(frame.get("voluntary_override"))
        if override:
            rep["override_ticks"] += 1
        else:
            node0, _ = compete(base, min_sal)
            rep["fidelity"]["total"] += 1
            rep["fidelity"]["agree"] += int(node0 == logged or (node0 is None and logged_action == "none"))
        for node, fn in values.items():
            c = rep["candidates"][node]
            pe = fn(t)
            if not pe:
                runs[node] = 0
                continue
            c["nonzero_ticks"] += 1
            c["max_value"] = max(c["max_value"], round(pe, 3))
            dyn = round(PE_TO_DYNAMIC * pe, 6)
            if dyn < min_sal:
                runs[node] = 0
                continue
            c["admitted_ticks"] += 1
            if override:
                runs[node] = 0
                continue
            winner, action = compete(base + [_node(node, CANDIDATES.get(node, node), dyn)], min_sal)
            if winner == node:
                c["wins"] += 1
                c["wins_with_action"] += int(action != "none")
                c["displaced"][classify(logged) if logged_action != "none" else "no_action"] += 1
                c["win_hours"][t.strftime("%Y-%m-%dT%H")] += 1
                runs[node] += 1
                c["longest_win_run"] = max(c["longest_win_run"], runs[node])
                # dwell_ticks >= 2 needs the coalition in 2 of 3 ticks (activation, dwell 1) and one
                # more tick (dwell 2): the third consecutive win is the first bindable one.
                c["bindable_ticks"] += int(runs[node] >= 3 and action != "none")
            else:
                runs[node] = 0
    for c in rep["candidates"].values():
        c["displaced"] = dict(c["displaced"].most_common())
        c["distinct_win_hours"] = len(c.pop("win_hours"))
        c["win_share"] = round(c["wins"] / rep["ticks"], 4) if rep["ticks"] else 0.0
    f = rep["fidelity"]
    f["rate"] = round(f["agree"] / f["total"], 4) if f["total"] else None
    return rep


def _load(conn, hours: float) -> tuple[list[dict], list[TempPoint], Series, Series]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    with conn.cursor() as cur:
        cur.execute("SELECT generated_at, projection_json FROM substrate_attention_broadcast_log "
                    "WHERE generated_at >= %s ORDER BY generated_at", (since,))
        rows = [{"generated_at": r[0], "projection": r[1]} for r in cur.fetchall()]
        cur.execute("SELECT timestamp::timestamptz, (measurements->>'cabinet_temp_c')::float FROM orion_biometrics_summary "
                    "WHERE node = 'athena' AND timestamp >= %s AND measurements ? 'cabinet_temp_c' ORDER BY timestamp",
                    ((since - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),))
        points = [TempPoint(r[0], r[1]) for r in cur.fetchall() if r[1] is not None]
        cur.execute("""SELECT DISTINCT ON (date_trunc('minute', created_at)) created_at,
                         (field_json->'node_vectors'->'node:substrate.rpc_delivery'->>'rpc_timeout_pressure')::float,
                         (field_json->'node_vectors'->'node:circe'->>'inference_failure_pressure')::float
                       FROM substrate_field_state WHERE created_at >= %s
                       ORDER BY date_trunc('minute', created_at), created_at DESC""", (since,))
        rpc, circe = Series(), Series()
        for ts, r, c in cur.fetchall():
            for s, v in ((rpc, r), (circe, c)):
                if v is not None:
                    s.ts.append(ts)
                    s.vals.append(max(0.0, min(1.0, v)))
    return rows, points, rpc, circe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=float, default=72.0)
    ap.add_argument("--rise-threshold-c", type=float, default=0.5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    dsn = os.environ.get("ORION_PG_DSN") or os.environ.get("POSTGRES_URI")
    if not dsn:
        print("ORION_PG_DSN or POSTGRES_URI required", file=sys.stderr)
        return 2
    import psycopg2

    with psycopg2.connect(dsn, options="-c default_transaction_read_only=on -c statement_timeout=120000") as conn:
        rows, points, rpc, circe = _load(conn, args.hours)
    cab = cabinet_series(points, [r["generated_at"] for r in rows], args.rise_threshold_c)
    rep = replay(rows, {"node:substrate.cabinet": cab.get, "node:substrate.rpc_delivery": rpc.at,
                        "node:circe": circe.at})
    rep["window_hours"] = args.hours
    rep["cabinet_samples"] = len(points)
    rep["rise_threshold_c"] = args.rise_threshold_c
    print(json.dumps(rep, indent=None if args.json else 2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
