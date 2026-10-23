"""Live smoke for the power_attribution introspect tool: real SQL, real tables, cross-checked.

Runs orion/introspect/power_attribution.py's three queries against live Postgres in a READ
ONLY transaction, runs the analysis, then checks it by an independent path:

* every circe step's before/after watts against a direct AVG over orion_energy_hourly for the
  same hours;
* when 2026-10-14 03:00 UTC is inside the window: an attribution within 6 h of it comes back
  (a wall step, or the CPU-load shift when hourly wander keeps the wall step under its bar),
  it credits a CPU-load regime with >= 60 W (hand-found: ~144 W), and the load jump
  (~1 -> ~13) shows up as a load_15m shift;
* the whole result fits the 12k MCP budget, and every node's text says UNVERIFIED when it
  makes a claim.

Prints each node's plain-English answer, exactly as Orion would get it.

  POSTGRES_URI=postgresql://... python services/orion-hub/evals/run_power_attribution_live_eval.py --days 14

Needs a live database, so it is not a CI step; the CI-safe checks are
orion/physiology/tests/test_heat_archaeology.py, orion/introspect/tests/test_power_attribution.py,
services/orion-hub/tests/test_power_attribution_introspect_listener.py and
services/orion-hub/evals/run_power_attribution_eval.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.introspect import power_attribution as pa  # noqa: E402
from orion.schemas.introspect import IntrospectResultV1, PowerAttributionArguments  # noqa: E402

KNOWN_STEP_AT = datetime(2026, 10, 14, 3, tzinfo=timezone.utc)
# Hand-found 2026-10-23: the crash loop was worth ~144 W; the fit must credit most of it.
KNOWN_REGIME_MIN_W = 60.0
KNOWN_STEP_SLACK = timedelta(hours=6)
MCP_BUDGET = 12000
DIRECT_AVG_SQL = """
select avg(kwh * 3600000.0 / covered_sec) as w, count(*) as n
from orion_energy_hourly
where source = $1 and hour_start >= $2 and hour_start <= $3 and covered_sec >= $4
"""


async def run(dsn: str, days: float, until: datetime | None) -> int:
    import asyncpg

    args = PowerAttributionArguments(days=days, until=until)
    now = datetime.now(timezone.utc)
    start, end = args.window(now)
    conn = await asyncpg.connect(dsn)
    failures: list[str] = []
    try:
        async with conn.transaction(readonly=True):
            results = await pa.read_power_attribution(conn, start, end)
            for r in results:
                hours = [s.at for s in r.steps]
                edges = [start, *hours, end]
                for i, step in enumerate(r.steps):
                    lo, hi = edges[i], edges[i + 2]
                    before = await conn.fetchrow(DIRECT_AVG_SQL, r.node, lo, step.at - timedelta(hours=1), pa.MIN_COVERED_SEC)
                    after = await conn.fetchrow(DIRECT_AVG_SQL, r.node, step.at, hi - timedelta(seconds=1), pa.MIN_COVERED_SEC)
                    for label, direct, tool in (("before", before, step.before_w), ("after", after, step.after_w)):
                        if direct["w"] is None or abs(float(direct["w"]) - tool) > 2.0:
                            failures.append(f"{r.node} step {step.at:%Y-%m-%d %H:%M} {label}: tool {tool} W "
                                            f"!= direct {direct['w']} W")
    finally:
        await conn.close()

    if start <= KNOWN_STEP_AT - timedelta(days=1) and KNOWN_STEP_AT + timedelta(days=1) <= end:
        circe = next((r for r in results if r.node == "circe"), None)
        # Either a wall step or, when hourly wander keeps the wall step under its bar, the
        # CPU-load shift's attribution must carry the 10-14 change.
        attr = None if circe is None else next(
            (a for a in circe.attributions if abs(a.step_at - KNOWN_STEP_AT) <= KNOWN_STEP_SLACK), None)
        if attr is None:
            failures.append("the 2026-10-14 circe change did not come back (no step or load-shift attribution)")
        else:
            regime = next((v for k, v in attr.contributions_w.items() if k.startswith("CPU-load regime")), None)
            if regime is None or regime < KNOWN_REGIME_MIN_W:
                failures.append(f"10-14 attribution's CPU-load regime term is {regime} W, under {KNOWN_REGIME_MIN_W}")
            if not any(s.series == "load_15m" and abs(s.at - KNOWN_STEP_AT) <= KNOWN_STEP_SLACK and s.delta > 5
                       for s in circe.regime_shifts):
                failures.append("10-14 load_15m jump not reported as a shift")
            print(f"# 10-14 ({attr.anchor}): {attr.delta_w:+.0f} W at {attr.step_at:%Y-%m-%d %H:%M} UTC, "
                  f"regime {regime} W, fit R2 {circe.fit.r2 if circe.fit else None}", file=sys.stderr)
    else:
        print("# 2026-10-14 is outside this window; the known-step check is skipped", file=sys.stderr)

    items = pa.build_items(results, start, end)
    body = json.dumps(IntrospectResultV1(
        ok=True, operation="power_attribution", as_of=now, total_available=len(items), items=items,
    ).model_dump(mode="json"), ensure_ascii=False)
    if len(body) > MCP_BUDGET:
        failures.append(f"result is {len(body)} chars, over the {MCP_BUDGET} MCP budget")
    for r in results:
        if (r.steps or r.fit) and "UNVERIFIED" not in r.summary:
            failures.append(f"{r.node}: a claim without UNVERIFIED")
        print(f"\n{r.summary}")
    print(f"\n# window {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC, result_chars={len(body)}", file=sys.stderr)
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    print("PASS" if not failures else f"FAILED ({len(failures)})", file=sys.stderr)
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=14)
    parser.add_argument("--until", type=datetime.fromisoformat, default=None)
    parser.add_argument("--dsn", default=os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL"))
    ns = parser.parse_args(argv)
    if not ns.dsn:
        print("POSTGRES_URI (or --dsn) is required: this eval reads live Postgres", file=sys.stderr)
        return 2
    return asyncio.run(run(ns.dsn, ns.days, ns.until))


if __name__ == "__main__":
    raise SystemExit(main())
