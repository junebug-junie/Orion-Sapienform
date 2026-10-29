#!/usr/bin/env python3
"""Fit CPU package watts ~ busy cores per host; print the idle floor (intercept) and slope.

The intercept is each node's CPU idle floor for the per-container CPU watt split
(services/orion-biometrics CPU_IDLE_WATTS_BY_NODE, orion/telemetry/biometrics_pipeline.py
container_cpu_measurements). Read-only: one SELECT through psql in a read-only transaction.

Busy cores = raw orion_biometrics.cpu.util x summary cpu_cores (joined on node + timestamp);
watts = summary measurements.cpu_watts_total (RAPL). Once `cpu_busy_cores` is in measurements
(2026-10-29), the join is no longer needed, but the raw join keeps the history usable.

    python3 scripts/analysis/fit_cpu_idle_watts.py --since 2026-10-15
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

DEFAULT_PSQL = "docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -At -q -v ON_ERROR_STOP=1"

SQL = """
with j as (
  select s.node, (b.cpu->>'util')::float * (s.measurements->>'cpu_cores')::float as busy,
         (s.measurements->>'cpu_watts_total')::float as w
    from orion_biometrics_summary s
    join orion_biometrics b on b.node = s.node and b.timestamp = replace(s.timestamp, ' ', 'T') || ':00'
   where s.timestamp >= '{since}' and s.measurements ? 'cpu_watts_total' and b.cpu ? 'util')
select coalesce(json_agg(r), '[]') from (
  select node, count(*) as n, regr_intercept(w, busy) as idle_w, regr_slope(w, busy) as w_per_core,
         regr_r2(w, busy) as r2, min(busy) as busy_min, max(busy) as busy_max, min(w) as w_min
    from j where busy > 0 group by node order by node) r;
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", required=True, help="UTC date or timestamp, e.g. 2026-10-15")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?", args.since):
        ap.error("--since must look like 2026-10-15 or 2026-10-15 00:00")
    cmd = shlex.split(os.environ.get("ORION_ANALYSIS_PSQL", DEFAULT_PSQL))
    sql = "begin read only;\n" + SQL.format(since=args.since.replace("T", " ")) + "\ncommit;\n"
    proc = subprocess.run(cmd, input=sql, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(proc.stderr.strip()[:500], file=sys.stderr)
        return 1
    rows = json.loads(proc.stdout.strip() or "[]")
    print(f"{'node':8} {'n':>7} {'idle_w':>7} {'W/core':>7} {'r2':>5} {'busy':>12} {'w_min':>6}")
    for r in rows:
        print(f"{r['node']:8} {r['n']:7d} {r['idle_w']:7.1f} {r['w_per_core']:7.2f} {r['r2']:5.2f} "
              f"{r['busy_min']:5.1f}-{r['busy_max']:5.1f} {r['w_min']:6.1f}")
    print('CPU_IDLE_WATTS_BY_NODE=' + json.dumps({r["node"]: round(r["idle_w"], 0) for r in rows},
                                                   separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
