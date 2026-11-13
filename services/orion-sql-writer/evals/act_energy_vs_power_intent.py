#!/usr/bin/env python3
"""Eval: the GPU part of per-act energy from 30 s card samples vs the 1 Hz power-intent meter.

WHAT THIS CHECKS (2026-10-30). The act ledger's ``wh_by_host`` is now the act's share of the WHOLE
MACHINE above rest (chassis watts at the wall). The power-intent meter is GPU-only (one card's
nvidia-smi watts above a pre-window sample), so it can only check the act meter's GPU step:
``wh_gpu_by_host``, the card's watts above its idle floor (orion/energy/act_energy.card_energy).
That is what this replays. The whole-machine number is checked by a different test, closure:
services/orion-sql-writer/evals/act_energy_closure.py (acts + unmetered == measured above rest).

The act ledger's GPU step charges an act the card's watts above its idle floor from the ~30 s
biometrics samples. The power-intent loop measures the same reverie images at ~1 Hz
(power_intent_settled). This replays the GPU step over each settled power-intent window and
compares:

* per run: the ratio estimator / meter. A 30 s sample can land on a peak or miss it, so single
  runs scatter widely. That is expected and is why `basis` names the sample count.
* per family: total Wh over all runs. This is the number a family's cost reads, and it must
  agree within the gate below (an unbiased estimator converges; a biased one does not).

Samples come from the raw `orion_biometrics.gpu` blob (per-card watts there predate the new
`gpu{N}_watts` summary key, and are the same nvidia-smi read). Read-only.

    python3 services/orion-sql-writer/evals/act_energy_vs_power_intent.py --days 7
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from orion.energy.act_energy import card_energy, idle_floor, incremental_joules  # noqa: E402

DEFAULT_PSQL = "docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -At -q -v ON_ERROR_STOP=1"
FAMILY_GATE = 0.35  # |family total ratio - 1| above this fails the eval

SQL = """
with runs as (
  select intent_id, node, gpu_index, window_start, window_end, actual_mean_watts, baseline_watts
    from power_intent_settled
   where workload_kind = 'reverie_diffusion' and outcome = 'settled' and gpu_index is not null
     and window_start >= now() - interval '{days} days' and baseline_watts is not null
   order by window_start desc limit {limit})
select coalesce(json_agg(json_build_object(
  'intent_id', r.intent_id, 'node', r.node, 'gpu', r.gpu_index,
  'start', r.window_start, 'end', r.window_end, 'mean', r.actual_mean_watts, 'baseline', r.baseline_watts,
  'samples', (select json_agg(json_build_object('ts', b.timestamp, 'w', trim(g->>'power_draw_watts')) order by b.timestamp)
                from orion_biometrics b, jsonb_array_elements(b.gpu->'gpus') g
               where b.node = r.node and trim(g->>'gpu_index') = r.gpu_index::text
                 and b.timestamp >= to_char(r.window_start - interval '120 seconds', 'YYYY-MM-DD"T"HH24:MI:SS')
                 and b.timestamp <= to_char(r.window_end + interval '120 seconds', 'YYYY-MM-DD"T"HH24:MI:SS')),
  'history', (select json_agg(trim(g->>'power_draw_watts'))
                from orion_biometrics b, jsonb_array_elements(b.gpu->'gpus') g
               where b.node = r.node and trim(g->>'gpu_index') = r.gpu_index::text
                 -- the card's idle floor is day-keyed: the 24 h before the window's UTC day
                 and b.timestamp >= to_char(date_trunc('day', r.window_start at time zone 'UTC') - interval '24 hours',
                                            'YYYY-MM-DD"T"HH24:MI:SS')
                 and b.timestamp < to_char(date_trunc('day', r.window_start at time zone 'UTC'), 'YYYY-MM-DD"T"HH24:MI:SS'))
)), '[]') from runs r;
"""


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def evaluate(runs: list[dict]) -> dict:
    rows = []
    for r in runs:
        start, end = _ts(r["start"]), _ts(r["end"])
        meter_j = incremental_joules(float(r["mean"]), float(r["baseline"]), (end - start).total_seconds())
        points = [(_ts(s["ts"]), float(s["w"])) for s in (r.get("samples") or []) if s.get("w")]
        idle = idle_floor([float(v) for v in (r.get("history") or []) if v])
        if meter_j is None or idle is None:
            continue
        got = card_energy(points, idle_w=idle, start=start, end=end)
        if got is None:
            continue
        rows.append({"intent_id": r["intent_id"], "meter_wh": meter_j / 3600.0, "act_wh": got.wh,
                     "samples": got.samples_in_window, "idle_w": idle, "baseline_w": float(r["baseline"])})
    if not rows:
        return {"n": 0}
    meter = sum(x["meter_wh"] for x in rows)
    act = sum(x["act_wh"] for x in rows)
    ratios = [x["act_wh"] / x["meter_wh"] for x in rows if x["meter_wh"] > 0]
    return {
        "n": len(rows),
        "meter_wh_total": meter,
        "act_wh_total": act,
        "family_ratio": act / meter if meter > 0 else None,
        "meter_wh_mean": meter / len(rows),
        "act_wh_mean": act / len(rows),
        "per_run_ratio_median": statistics.median(ratios) if ratios else None,
        "per_run_ratio_p10_p90": (statistics.quantiles(ratios, n=10)[0], statistics.quantiles(ratios, n=10)[-1])
        if len(ratios) >= 10 else None,
        "idle_minus_baseline_w_mean": statistics.mean(x["idle_w"] - x["baseline_w"] for x in rows),
        "runs_with_zero_samples_in_window": sum(1 for x in rows if x["samples"] == 0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args(argv)
    cmd = shlex.split(os.environ.get("ORION_ANALYSIS_PSQL", DEFAULT_PSQL))
    sql = "begin read only;\n" + SQL.format(days=int(args.days), limit=int(args.limit)) + "\ncommit;\n"
    proc = subprocess.run(cmd, input=sql, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        print(proc.stderr.strip()[:500], file=sys.stderr)
        return 2
    out = evaluate(json.loads(proc.stdout.strip() or "[]"))
    print(json.dumps(out, indent=2, default=str))
    if not out.get("n"):
        print("UNVERIFIED: no comparable runs", file=sys.stderr)
        return 2
    ok = abs(out["family_ratio"] - 1.0) <= FAMILY_GATE
    print(f"family gate |ratio-1| <= {FAMILY_GATE}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
