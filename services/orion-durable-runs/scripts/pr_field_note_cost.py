#!/usr/bin/env python3
"""Measured cost of Orion's PR field notes (spec docs/superpowers/specs/2026-11-04-pr-field-notes-design.md,
"Cost per note" and acceptance check 6). Replaces the spec's ESTIMATE table with numbers from the rail.

Also the daily reminiscence (``--workflow reminiscence.episode``; design
docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md section 7, "ESTIMATE until Patch 2 measures 7
episodes"): the same joins, one run a day with six turns (five hops and one cross-read), all counted.

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
    python3 services/orion-durable-runs/scripts/pr_field_note_cost.py --workflow reminiscence.episode

Stdlib only; reads Postgres with ``docker exec <container> psql`` (read-only SELECT). Exit 0 always
when the query ran (even with no runs yet: it says so); 2 when psql failed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys

SQL_TEMPLATE = r"""
WITH runs AS (
  SELECT r.run_id, r.terminal, (r.request->'brief'->>'pr_number')::int AS pr,
         r.request->'brief'->>'episode' AS episode
  FROM durable_admission_runs r
  WHERE r.request->>'workflow' = '%(workflow)s' AND r.terminal IS NOT NULL
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
  FROM grammar_acts a WHERE a.actor = '%(actor)s' AND a.closed_at IS NOT NULL
)
SELECT json_build_object(
  'run_id', runs.run_id, 'pr', runs.pr, 'episode', runs.episode, 'terminal', runs.terminal,
  'turns', tok.turns, 'llm_calls', tok.calls, 'prompt', tok.prompt, 'cached', tok.cached,
  'completion', tok.completion, 'harness_steps', steps.harness_steps,
  'seconds', (act.cost_json->>'seconds')::float,
  'wh_by_host', act.cost_json->'wh_by_host', 'gpu_slot_s_by_lane', act.cost_json->'gpu_slot_s_by_lane',
  'usd', (act.cost_json->>'usd')::float)
FROM runs LEFT JOIN tok USING (run_id) LEFT JOIN steps USING (run_id) LEFT JOIN act USING (run_id)
ORDER BY runs.run_id
"""

# workflow -> (act-ledger actor, runs a day for the projection, completed runs the design asks for, unit)
WORKFLOWS = {
    "pr.field_note": ("durable.pr_field_note", 24.0, 20, "notes"),          # field-notes acceptance check 6
    "reminiscence.episode": ("durable.reminiscence", 1.0, 7, "episodes"),   # reminiscence design section 7
}


def sql_for(workflow: str, days: int) -> str:
    actor = WORKFLOWS[workflow][0]
    return SQL_TEMPLATE.replace("%(workflow)s", workflow).replace("%(actor)s", actor).replace("%(days)s", str(int(days)))


# The field-note query, unchanged in meaning (kept as a name: tests and the spec cite it).
SQL = SQL_TEMPLATE.replace("%(workflow)s", "pr.field_note").replace("%(actor)s", "durable.pr_field_note")

METRICS = (("seconds", "wall s (first grant -> end)"), ("wh", "Wh, whole machine above rest"),
           ("gpu_slot_s", "GPU slot s"), ("prompt", "prompt tokens"), ("cached", "cached prompt tokens"),
           ("completion", "completion tokens"), ("turns", "turns (attempts)"), ("harness_steps", "Hub harness steps"))


def query(days: int, container: str, db: str, workflow: str = "pr.field_note") -> list[dict]:
    sql = sql_for(workflow, days)
    proc = subprocess.run(["docker", "exec", container, "psql", "-U", "postgres", "-d", db, "-At", "-c", sql],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:500])
    return [with_sums(json.loads(ln)) for ln in proc.stdout.splitlines() if ln.strip()]


def map_sum(value: object) -> float | None:
    """Sum of a per-host / per-lane map. The act ledger writes JSON ``null`` when it could not attribute
    the act (seen live on 2026-11-06: one act of 46). That reads as UNREAD (None), never as 0, and never
    crashes the whole report the way summing it in SQL did (``jsonb_each_text`` on a non-object)."""
    if not isinstance(value, dict) or not value:
        return None
    total = 0.0
    for v in value.values():  # one unreadable entry makes the whole sum UNREAD, never a partial number
        if isinstance(v, bool) or v is None:
            return None
        try:
            total += float(v)
        except (TypeError, ValueError):
            return None
    return total


def with_sums(row: dict) -> dict:
    row["wh"] = map_sum(row.pop("wh_by_host", None))
    row["gpu_slot_s"] = map_sum(row.pop("gpu_slot_s_by_lane", None))
    return row


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def summarize(rows: list[dict], per_day: float, workflow: str = "pr.field_note") -> list[str]:
    _actor, _per_day, target, unit = WORKFLOWS[workflow]
    done = [r for r in rows if r["terminal"] == "completed"]
    out = [f"{workflow} runs ended: {len(rows)} ({len(done)} completed); medians/p90 over completed runs"]
    if not done:
        return out + ["no completed runs yet: the spec's cost table is still an ESTIMATE"]
    for key, label in METRICS:
        vals = [float(r[key]) for r in done if r.get(key) is not None]
        if not vals:
            out.append(f"  {label:34s} UNREAD (no rows joined yet)")
            continue
        med = statistics.median(vals)
        out.append(f"  {label:34s} median {med:10.1f}  p90 {pct(vals, 0.9):10.1f}  n={len(vals)}"
                   + (f"  -> ~{med * per_day:,.0f}/day at {per_day:g} {unit}"
                      if key in ("seconds", "wh", "prompt", "gpu_slot_s") else ""))
    if len(done) < target:
        out.append(f"  ({len(done)} of the {target} {unit} the design asks for)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--workflow", choices=sorted(WORKFLOWS), default="pr.field_note")
    ap.add_argument("--per-day", type=float, default=None,
                    help="runs per day for the projection (default: 24 notes, 1 reminiscence)")
    ap.add_argument("--container", default="orion-athena-sql-db")
    ap.add_argument("--db", default="conjourney")
    ap.add_argument("--rows", action="store_true", help="one JSON line per run")
    ap.add_argument("--print-sql", action="store_true")
    args = ap.parse_args(argv)
    per_day = WORKFLOWS[args.workflow][1] if args.per_day is None else args.per_day
    if args.print_sql:
        print(sql_for(args.workflow, args.days))
        return 0
    try:
        rows = query(args.days, args.container, args.db, args.workflow)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"psql failed: {exc}", file=sys.stderr)
        return 2
    if args.rows:
        for r in rows:
            print(json.dumps(r))
    print("\n".join(summarize(rows, per_day, args.workflow)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
