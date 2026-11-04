#!/usr/bin/env python3
"""Measured cost of Orion's PR field notes (spec docs/superpowers/specs/2026-11-04-pr-field-notes-design.md,
"Cost per note" and acceptance check 6). Replaces the spec's ESTIMATE table with numbers from the rail.

Per ended ``pr.field_note`` run, joined from what is already stored (nothing new is recorded for this):

* wall seconds, whole-machine Wh above rest (``wh_by_host``, not GPU-only), GPU slot seconds by lane:
  the run's act in ``grammar_acts`` (actor ``durable.pr_field_note``, ``object_ref durable_run:<run>``),
  filled by the act ledger's per-act attribution (PR #2864 and the whole-machine split);
* prompt / cached / completion tokens: ``llm_call_usage`` rows of EVERY step turn of the run (each
  attempt's ``step_correlation_id`` is on its ``run.started`` event in ``durable_resource_events``), so
  a retried note pays for its failed attempts too;
* turns and Hub harness steps: the same turn ids, ``harness_turn_trace.run_artifact.step_count``.

    python3 services/orion-durable-runs/scripts/pr_field_note_cost.py            # median / p90 table
    python3 services/orion-durable-runs/scripts/pr_field_note_cost.py --rows     # one line per run
    python3 services/orion-durable-runs/scripts/pr_field_note_cost.py --print-sql

Stdlib only; reads Postgres with ``docker exec <container> psql`` (read-only SELECT). Exit 0 always
when the query ran (even with no runs yet: it says so); 2 when psql failed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys

SQL = r"""
WITH runs AS (
  SELECT r.run_id, r.terminal, (r.request->'brief'->>'pr_number')::int AS pr
  FROM durable_admission_runs r
  WHERE r.request->>'workflow' = 'pr.field_note' AND r.terminal IS NOT NULL
    AND r.created_at >= now() - make_interval(days => %(days)s)
), turns AS (
  SELECT DISTINCT e.run_id, e.payload->'detail'->>'step_correlation_id' AS corr
  FROM durable_resource_events e JOIN runs USING (run_id)
  WHERE e.event = 'run.started' AND e.payload->'detail'->>'step_correlation_id' IS NOT NULL
), tok AS (
  SELECT t.run_id, count(DISTINCT t.corr) AS turns,
         coalesce(sum(u.prompt_tokens), 0) AS prompt, coalesce(sum(u.cached_prompt_tokens), 0) AS cached,
         coalesce(sum(u.completion_tokens), 0) AS completion, count(u.call_id) AS calls
  FROM turns t LEFT JOIN llm_call_usage u ON u.correlation_id = t.corr GROUP BY t.run_id
), steps AS (
  SELECT t.run_id, sum((h.run_artifact->>'step_count')::int) AS harness_steps
  FROM turns t JOIN harness_turn_trace h ON h.correlation_id = t.corr GROUP BY t.run_id
), act AS (
  SELECT replace(a.object_ref, 'durable_run:', '') AS run_id, a.cost_json
  FROM grammar_acts a WHERE a.actor = 'durable.pr_field_note' AND a.closed_at IS NOT NULL
)
SELECT json_build_object(
  'run_id', runs.run_id, 'pr', runs.pr, 'terminal', runs.terminal,
  'turns', tok.turns, 'llm_calls', tok.calls, 'prompt', tok.prompt, 'cached', tok.cached,
  'completion', tok.completion, 'harness_steps', steps.harness_steps,
  'seconds', (act.cost_json->>'seconds')::float,
  'wh', (SELECT sum(value::float) FROM jsonb_each_text(coalesce(act.cost_json->'wh_by_host', '{}'::jsonb))),
  'gpu_slot_s', (SELECT sum(value::float) FROM jsonb_each_text(coalesce(act.cost_json->'gpu_slot_s_by_lane', '{}'::jsonb))),
  'usd', (act.cost_json->>'usd')::float)
FROM runs LEFT JOIN tok USING (run_id) LEFT JOIN steps USING (run_id) LEFT JOIN act USING (run_id)
ORDER BY runs.run_id
"""

METRICS = (("seconds", "wall s (first grant -> end)"), ("wh", "Wh, whole machine above rest"),
           ("gpu_slot_s", "GPU slot s"), ("prompt", "prompt tokens"), ("cached", "cached prompt tokens"),
           ("completion", "completion tokens"), ("turns", "turns (attempts)"), ("harness_steps", "Hub harness steps"))


def query(days: int, container: str, db: str) -> list[dict]:
    sql = SQL.replace("%(days)s", str(int(days)))
    proc = subprocess.run(["docker", "exec", container, "psql", "-U", "postgres", "-d", db, "-At", "-c", sql],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:500])
    return [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def summarize(rows: list[dict], per_day: float) -> list[str]:
    done = [r for r in rows if r["terminal"] == "completed"]
    out = [f"pr.field_note runs ended: {len(rows)} ({len(done)} completed); medians/p90 over completed runs"]
    if not done:
        return out + ["no completed runs yet: the spec's cost table is still an ESTIMATE"]
    for key, label in METRICS:
        vals = [float(r[key]) for r in done if r.get(key) is not None]
        if not vals:
            out.append(f"  {label:34s} UNREAD (no rows joined yet)")
            continue
        med = statistics.median(vals)
        out.append(f"  {label:34s} median {med:10.1f}  p90 {pct(vals, 0.9):10.1f}  n={len(vals)}"
                   + (f"  -> ~{med * per_day:,.0f}/day at {per_day:g} notes" if key in ("seconds", "wh", "prompt") else ""))
    if len(done) < 20:
        out.append(f"  ({len(done)} of the 20 notes acceptance check 6 asks for)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--per-day", type=float, default=24.0, help="notes per day for the projection")
    ap.add_argument("--container", default="orion-athena-sql-db")
    ap.add_argument("--db", default="conjourney")
    ap.add_argument("--rows", action="store_true", help="one JSON line per run")
    ap.add_argument("--print-sql", action="store_true")
    args = ap.parse_args(argv)
    if args.print_sql:
        print(SQL.replace("%(days)s", str(args.days)))
        return 0
    try:
        rows = query(args.days, args.container, args.db)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"psql failed: {exc}", file=sys.stderr)
        return 2
    if args.rows:
        for r in rows:
            print(json.dumps(r))
    print("\n".join(summarize(rows, args.per_day)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
