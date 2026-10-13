"""Live eval for the node_history introspect tool: real SQL, real table, checked against raw rows.

Runs orion/introspect/node_history.py's queries against live Postgres in a READ ONLY
transaction, then pulls the window's raw rows and recomputes every summary field in Python.
Fails if any field disagrees, if the series has more than 48 points, or if the answer would
not fit the 12k MCP result budget. Prints the item exactly as Orion would get it.

  POSTGRES_URI=postgresql://... python services/orion-hub/evals/run_node_history_live_eval.py \
      --node node:substrate.bus_synaptic --hours 72

Needs a live database, so it is not a CI step; the CI-safe checks are
orion/introspect/tests/test_node_history.py and the Hub listener tests.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.introspect import node_history as nh  # noqa: E402
from orion.schemas.introspect import NODE_HISTORY_MAX_POINTS, IntrospectResultV1, NodeHistoryArguments  # noqa: E402

RAW_SQL = f"select observed_at, value from {nh.TABLE} where node_id = $1 and observed_at > $2 and observed_at <= $3 order by observed_at"
MCP_BUDGET = 12000


def _close(a, b) -> bool:
    return a == b or (a is not None and b is not None and abs(a - b) <= 1e-12 + 1e-9 * abs(b))


async def run(dsn: str, node: str, hours: float) -> int:
    import asyncpg

    args = NodeHistoryArguments(node_id=node, hours=hours)
    now = datetime.now(timezone.utc)
    start, end = args.window(now)
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction(readonly=True):
            item = await nh.read_node_history(conn, node, start, end)
            raw = await conn.fetch(RAW_SQL, node, start, end)
    finally:
        await conn.close()
    failures: list[str] = []
    if item is None:
        print(json.dumps({"node": node, "hours": hours, "items": []}))
        if raw:
            failures.append(f"tool said no readings but {len(raw)} raw rows exist")
    else:
        ex = item.extra
        values = [float(r["value"]) for r in raw]
        expected = {
            "count": len(values), "min": nh.sig(min(values)), "max": nh.sig(max(values)),
            "median": nh.sig(statistics.median(values)),
        }
        for key, want in expected.items():
            if not _close(ex[key], want):
                failures.append(f"{key}: tool {ex[key]} != raw {want}")
        if ex["first"]["value"] != nh.sig(values[0]) or ex["last"]["value"] != nh.sig(values[-1]):
            failures.append("first/last value disagree with raw rows")
        if len(ex["series"]) > NODE_HISTORY_MAX_POINTS:
            failures.append(f"series has {len(ex['series'])} points")
        if sum(p[1] for p in ex["series"]) != len(values):
            failures.append("series bucket counts do not add up to the reading count")
        deltas = sorted((abs(b - a) for a, b in zip(values, values[1:])), reverse=True)
        if deltas and not _close(abs(ex["largest_changes"][0]["delta"]), nh.sig(deltas[0])):
            failures.append(f"largest change {ex['largest_changes'][0]['delta']} != raw {deltas[0]}")
        result = IntrospectResultV1(ok=True, operation="node_history", as_of=now, total_available=1, items=[item])
        body = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
        if len(body) > MCP_BUDGET:
            failures.append(f"result is {len(body)} chars, over the {MCP_BUDGET} MCP budget")
        print(body)
        print(f"# result_chars={len(body)} raw_rows={len(values)}", file=sys.stderr)
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    print("PASS" if not failures else f"FAILED ({len(failures)})", file=sys.stderr)
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="node:substrate.bus_synaptic")
    parser.add_argument("--hours", type=float, default=72)
    parser.add_argument("--dsn", default=os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL"))
    ns = parser.parse_args(argv)
    if not ns.dsn:
        print("POSTGRES_URI (or --dsn) is required: this eval reads live Postgres", file=sys.stderr)
        return 2
    return asyncio.run(run(ns.dsn, ns.node, ns.hours))


if __name__ == "__main__":
    raise SystemExit(main())
