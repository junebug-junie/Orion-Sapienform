"""24 h live sanity check for the PE-magnitude history (spec 2026-10-02, step 1).

Run this after ``SUBSTRATE_PE_HISTORY_ENABLED=true`` has been live for 24 h, and
before step 2 (the reverie consumer) ships. It covers the metric gate's
"live-data sanity" item for every domain that wins the broadcast today. It
replays ``compute_prediction_error_magnitude`` over the stored history every
``--step-min`` minutes of the last 24 h, then checks each node:

- ``coverage``: at least 200 readings in 7 days, otherwise band/trend stay
  ``insufficient_history``.
- ``not_flat``: more than one distinct value.
- ``trend_not_degenerate``: no single trend label above 90% of replay points.
- ``present``: all 9 domains that win the broadcast have rows.

It also reports, without gating, the minimum reading and the share of readings
at that minimum (can it return to rest?), plus the band distribution.

Read-only. Exit status is 1 if any check fails.

Run:
  POSTGRES_URI=postgresql://... python services/orion-substrate-runtime/evals/run_pe_magnitude_live_sanity.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.substrate.prediction_error_magnitude import (  # noqa: E402
    DEFAULT_MIN_READINGS,
    WINDOW_7D,
    compute_prediction_error_magnitude,
)

# Domains that won the broadcast in the 7 days before 2026-10-02 (spec, live evidence).
EXPECTED_NODES = tuple(
    f"node:substrate.{d}"
    for d in (
        "execution", "chat", "bus_synaptic", "biometrics", "route",
        "harness_closure", "perception", "codebase", "cabinet",
    )
)
MAX_TREND_SHARE = 0.90


def evaluate(
    rows: list[tuple[str, datetime, float]],
    *,
    now: datetime,
    step: timedelta = timedelta(minutes=10),
    trend_min_delta: float = 0.01,
    expected_nodes: tuple[str, ...] = EXPECTED_NODES,
) -> dict:
    """Pure: history rows -> per-node report + overall pass flag."""
    by_node: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for node_id, ts, value in rows:
        ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        by_node[node_id].append((ts, float(value)))

    report: dict[str, dict] = {}
    ok = True
    for node_id in sorted(set(by_node) | set(expected_nodes)):
        hist = sorted(by_node.get(node_id, []))
        in_7d = [v for ts, v in hist if ts >= now - WINDOW_7D]
        entry: dict = {"n_7d": len(in_7d), "checks": {}}
        entry["checks"]["present"] = bool(in_7d) or node_id not in expected_nodes
        if in_7d:
            lo = min(in_7d)
            entry["min"] = round(lo, 6)
            entry["share_at_min"] = round(sum(1 for v in in_7d if v == lo) / len(in_7d), 4)
            entry["distinct"] = len(set(in_7d))
        entry["checks"]["coverage"] = len(in_7d) >= DEFAULT_MIN_READINGS
        entry["checks"]["not_flat"] = entry.get("distinct", 0) > 1

        trends: Counter[str] = Counter()
        bands: Counter[str] = Counter()
        t = now - timedelta(hours=24)
        while t <= now:
            upto = [(ts, v) for ts, v in hist if ts <= t]
            if upto:
                ts_last, v_last = upto[-1]
                mag = compute_prediction_error_magnitude(
                    value=v_last, observed_at=ts_last, history=upto, now=t,
                    trend_min_delta=trend_min_delta,
                )
                trends[mag.trend] += 1
                bands[mag.band] += 1
            t += step
        total = sum(trends.values())
        entry["trend"] = dict(trends)
        entry["band"] = dict(bands)
        top_share = (trends.most_common(1)[0][1] / total) if total else 1.0
        entry["checks"]["trend_not_degenerate"] = total > 0 and top_share <= MAX_TREND_SHARE
        if not all(entry["checks"].values()):
            ok = False
        report[node_id] = entry
    return {"ok": ok, "nodes": report}


def _fetch(uri: str, since: datetime) -> list[tuple[str, datetime, float]]:
    from sqlalchemy import create_engine, text

    engine = create_engine(uri)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT node_id, observed_at, value FROM substrate_node_prediction_error_history "
                "WHERE observed_at >= :since ORDER BY node_id, observed_at"
            ),
            {"since": since},
        ).fetchall()
    return [(str(r[0]), r[1], float(r[2])) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--postgres-uri", default=os.getenv("POSTGRES_URI"))
    ap.add_argument("--step-min", type=float, default=10.0)
    ap.add_argument("--trend-min-delta", type=float,
                    default=float(os.getenv("ORION_REVERIE_PE_TREND_MIN_DELTA", "0.01")))
    args = ap.parse_args()
    if not args.postgres_uri:
        print("POSTGRES_URI / --postgres-uri required", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    rows = _fetch(args.postgres_uri, now - WINDOW_7D)
    result = evaluate(rows, now=now, step=timedelta(minutes=args.step_min),
                      trend_min_delta=args.trend_min_delta)
    for node_id, e in result["nodes"].items():
        failed = [k for k, v in e["checks"].items() if not v]
        print(
            f"{'FAIL' if failed else 'ok  '} {node_id:36s} n7d={e['n_7d']:6d} "
            f"distinct={e.get('distinct', 0):5d} min={e.get('min')} "
            f"at_min={e.get('share_at_min')} trend={e['trend']} band={e['band']}"
            + (f" failed={failed}" if failed else "")
        )
    print("PASS" if result["ok"] else "FAIL")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
