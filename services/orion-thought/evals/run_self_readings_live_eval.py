"""Live replay of the self-readings block against the real history table (read-only).

Acceptance checks 4 and 5 (spec 2026-10-07): at real moments from the last day when a body
signal was high, and right now, render the block exactly as orion-thought would, check every
number against the persisted rows (``self_readings_truth.check_block``), and report the prompt
chars and the read+render cost.

    python services/orion-thought/evals/run_self_readings_live_eval.py [--moments 5]

DSN: ``SELF_READINGS_EVAL_DSN`` (default the host-published athena Postgres, localhost:55432).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE))

from sqlalchemy import create_engine, text  # noqa: E402

from orion.attention.pe_history_cache import NodePeHistoryCache  # noqa: E402
from orion.attention.self_readings import prompt_reaching_nodes, project_self_readings  # noqa: E402
from self_readings_truth import check_block  # noqa: E402

_SQL = ("SELECT node_id, observed_at, value FROM substrate_node_prediction_error_history "
        "WHERE observed_at >= :since AND observed_at <= :until ORDER BY observed_at")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moments", type=int, default=5)
    args = ap.parse_args()
    dsn = os.getenv("SELF_READINGS_EVAL_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney")
    eng = create_engine(dsn)
    nodes = prompt_reaching_nodes()

    def fetcher(until: datetime):
        def fetch(since: datetime):
            with eng.connect() as c:
                return [(str(r[0]), r[1], float(r[2]))
                        for r in c.execute(text(_SQL), {"since": since, "until": until})]
        return fetch

    with eng.connect() as c:
        highs = [r[0] for r in c.execute(text(
            "SELECT DISTINCT ON (date_trunc('hour', observed_at)) observed_at "
            "FROM substrate_node_prediction_error_history "
            "WHERE observed_at > now() - interval '24 hours' AND node_id = ANY(:nodes) "
            "AND ((node_id IN ('node:substrate.execution','node:substrate.chat','node:substrate.codebase') "
            "AND value >= 0.9) OR (node_id = 'node:substrate.bus_synaptic' AND value >= 0.1) "
            "OR (node_id = 'node:substrate.cabinet' AND value > 0)) "
            "ORDER BY date_trunc('hour', observed_at) DESC, observed_at DESC LIMIT :k"),
            {"nodes": nodes, "k": args.moments})]
    moments = [("now", datetime.now(timezone.utc))] + [("replay", t + timedelta(seconds=20)) for t in highs]

    failures_total = 0
    rendered_chars: list[int] = []
    seed_ms: list[float] = []
    incr_ms: list[float] = []
    for kind, now in moments:
        fetch = fetcher(now)
        cache = NodePeHistoryCache()
        t0 = time.perf_counter()
        cache.refresh(fetch, now=now)
        block = project_self_readings(cache, now=now)
        seed_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        cache.refresh(fetch, now=now)
        project_self_readings(cache, now=now)
        incr_ms.append((time.perf_counter() - t1) * 1000)
        rows = fetch(now - timedelta(days=7, minutes=5))
        fails = check_block(block.text, rows, now)
        failures_total += len(fails)
        if block.text:
            rendered_chars.append(len(block.text))
        print(f"\n== {kind} {now.isoformat()} skip={block.skip_reason} chars={len(block.text)} "
              f"truth_failures={len(fails)}")
        for r in block.readings:
            print(f"   {r.source_id:30s} {r.state:20s} value={r.value} band={r.band} "
                  f"score={r.score_0_10} unusual={r.unusual}")
        if block.text:
            print("   ---\n   " + block.text.replace("\n", "\n   "))
        for f in fails:
            print("   FAIL", f)
    print(f"\nmoments={len(moments)} rendered={len(rendered_chars)} truth_failures={failures_total}")
    if rendered_chars:
        print(f"block chars p50={statistics.median(rendered_chars)} max={max(rendered_chars)}")
    print(f"seed+render ms p50={statistics.median(seed_ms):.0f} max={max(seed_ms):.0f}; "
          f"incremental+render ms p50={statistics.median(incr_ms):.1f} max={max(incr_ms):.1f}")
    return 1 if failures_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
