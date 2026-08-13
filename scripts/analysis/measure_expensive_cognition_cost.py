#!/usr/bin/env python3
"""Phase E0 cost census -- what does Orion's expensive cognition actually cost?

## Why this exists

`docs/superpowers/specs/2026-08-13-scarcity-and-repertoire-execution-plan.md`
Phase E0. The whole plan rests on one number nobody has ever measured.

`goal_formulate`, `dream_cycle`, `counterfactual`, and
`context_exec_memory_contradiction_review` are all defined in
`orion/cognition/verbs/`, all GPU-backed, and have **never once been executed**
by anything -- confirmed against 12h of live cortex-exec logs, where the only
verbs that appear are substrate.inspect/summarize/observe, journal.compose,
log_orion_metacognition and builder_prune.

So nobody knows whether a goal formulation takes 2 seconds or 90. That number
decides whether serialized inference time is a real scarce resource (and the
plan proceeds) or an invented one (and the plan dies).

## The two kill gates this script feeds

    KILL A: expensive cognition costs < 5s wall-clock
            -> compute is not scarce, thesis dead, Phases 1-5 cancelled.

    KILL B: fewer than 3 of 10 goal_formulate outputs are things a human
            would not have written
            -> the faculty is hollow, Phase 2 has nothing worth buying.

This script measures gate A and produces the verbatim material for a human to
judge gate B. It deliberately does NOT score gate B itself: an LLM grading
whether another LLM's output is generic is exactly the "two untested things
agreeing" failure the plan's own throttle rules ban.

## Disclosed scoping decisions

1. **Same path production would use.** Invokes through
   `ExecutionDispatchCortexClient` over the real bus, on the real background
   lane (`orion:cortex:exec:request:background`) -- byte-identical to how the
   arena would dispatch these verbs if they were routed. A cheaper in-process
   call would measure a different thing than the one the plan prices.

2. **Real field state, not synthetic.** Context is built from real
   `substrate_field_state` rows, so the model is reasoning about Orion's
   actual condition. A synthetic prompt would produce a synthetic cost.

3. **Sequential, never concurrent.** The resource under test IS serialized
   inference time. Running these in parallel would measure throughput under
   contention and understate per-action wall-clock, which is the exact
   quantity the plan needs.

4. **Wall-clock is the headline, tokens are secondary.** The plan's scarce
   resource is time against a tick deadline (~2.7s), not tokens. Tokens are
   recorded because they are the already-live meter Phase 1 would use.

5. **Read-only against Postgres**; the same session-level read-only enforcement
   as the sibling analysis scripts. This script does not write to any Orion
   store. It DOES cause real inference to run -- that is the measurement -- and
   any artifact those verbs persist internally is a real side effect, disclosed
   here rather than assumed absent.

6. **Bounded blast radius on the shared lane.** The background lane is shared
   with real dispatch. `--sleep-sec` (default 2.0) spaces requests so this
   census does not saturate the lane it is measuring.

## Usage

    ORION_BUS_URL=redis://100.92.216.81:6379/0 \\
    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_expensive_cognition_cost.py --runs 10

Exit codes: 0 = census completed (whatever it found -- this is an instrument,
and reaching a kill gate is a successful run), 1 = could not measure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Sequence

logger = logging.getLogger("measure_expensive_cognition_cost")

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@localhost:55432/conjourney"
DEFAULT_BUS_URL = "redis://100.92.216.81:6379/0"
DEFAULT_REQUEST_CHANNEL = "orion:cortex:exec:request:background"
DEFAULT_RESULT_PREFIX = "orion:exec:result"

# The four never-executed verbs named in the plan. Each is defined in
# orion/cognition/verbs/ and GPU-backed.
E0_VERBS = (
    "goal_formulate",
    "counterfactual",
    "dream_cycle",
    "context_exec_memory_contradiction_review",
)

# The plan's kill gate A, in one place so the report and the verdict cannot
# disagree about it.
KILL_GATE_A_MIN_WALL_SEC = 5.0

# The arena's real tick cadence, measured live: ~22 policy frames/min.
# Used only to express opportunity cost in ticks, which is what makes the
# scarcity argument concrete.
ARENA_TICK_SEC = 2.7


@dataclass
class RunResult:
    verb: str
    run_index: int
    tick_id: str | None
    wall_sec: float
    ok: bool
    error: str | None = None
    completion_tokens: int | None = None
    plan_status: str | None = None
    final_text: str = ""


@dataclass
class VerbSummary:
    verb: str
    runs: list[RunResult] = dataclass_field(default_factory=list)

    @property
    def ok_runs(self) -> list[RunResult]:
        return [r for r in self.runs if r.ok]

    @property
    def wall_times(self) -> list[float]:
        return [r.wall_sec for r in self.ok_runs]

    def stat(self, fn) -> float | None:
        times = self.wall_times
        return fn(times) if times else None


# ===========================================================================
# Field state -- real context, not synthetic
# ===========================================================================


def open_readonly_connection(dsn: str):
    try:
        import psycopg2
    except Exception:
        logger.error("psycopg2 unavailable; cannot read field state")
        return None
    try:
        conn = psycopg2.connect(dsn)
    except Exception:
        logger.error("failed to connect to postgres", exc_info=True)
        return None
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on;")
            cur.execute("SHOW default_transaction_read_only;")
            value = cur.fetchone()
        if not value or str(value[0]).lower() != "on":
            logger.error("refusing to run: session is not read-only (got %r)", value)
            conn.close()
            return None
    except Exception:
        logger.error("failed to enforce read-only session", exc_info=True)
        try:
            conn.close()
        except Exception:
            pass
        return None
    return conn


def load_real_field_ticks(conn, limit: int) -> list[dict[str, Any]]:
    """The most recent real field ticks, spread out rather than adjacent.

    Adjacent ticks are near-identical (the field moves slowly), so ten
    consecutive rows would ask the model the same question ten times and
    measure caching rather than cognition. Sampled across a window instead.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tick_id, field_json
            FROM (
                SELECT tick_id, field_json, generated_at,
                       row_number() OVER (ORDER BY generated_at DESC) AS rn
                FROM substrate_field_state
                ORDER BY generated_at DESC
                LIMIT 2000
            ) s
            WHERE rn %% (2000 / %s) = 1
            LIMIT %s
            """,
            (max(1, limit), limit),
        )
        rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for tick_id, field_json in rows:
        if isinstance(field_json, str):
            try:
                field_json = json.loads(field_json)
            except Exception:
                continue
        if isinstance(field_json, dict):
            out.append({"tick_id": tick_id, "field": field_json})
    return out


def summarise_field_as_raw_material(field: dict[str, Any]) -> str:
    """Orion's own live state, rendered as raw material -- NOT as a goal.

    The distinction matters and is the whole validity of gate B.

    The first smoke run exposed why: `goal_formulate`'s prompt template
    (`orion/cognition/prompts/goal_formulate_prompt.j2`) reads
    `{{ intention or text or request }}`. Passing only `field_state` left that
    empty, the template rendered "No raw intention provided", and the model --
    with recall enabled -- filled the vacuum from an unrelated past coding
    session. The first goal Orion ever formulated was about fixing Bash
    permissions on someone else's PR.

    So the verb is a TRANSLATOR (intention -> structured goal), not a
    GENERATOR (state -> intention). Measuring it with an empty intention slot
    would measure recall confabulation, not the faculty.

    This function therefore supplies Orion's real readings and nothing else:
    pressure values, anomaly z-scores, perturbation counts. No verb, no
    objective, no "you should" -- if a goal appears in the output, the model
    formed it. Whether that is enough to count as wanting is exactly what a
    human judges at gate B.
    """
    lines: list[str] = ["My current internal readings:"]

    zscores = field.get("dimension_precision_zscore")
    if isinstance(zscores, dict) and zscores:
        for dim, value in sorted(zscores.items()):
            if isinstance(value, (int, float)):
                lines.append(f"  {dim}: surprise z-score {float(value):+.3f}")

    ewma = field.get("dimension_precision_ewma")
    if isinstance(ewma, dict) and ewma:
        for dim, value in sorted(ewma.items()):
            if isinstance(value, (int, float)):
                lines.append(f"  {dim}: recent baseline {float(value):.4f}")

    perturbations = field.get("recent_perturbations")
    if isinstance(perturbations, (int, float)):
        lines.append(f"  recent perturbations: {perturbations}")
    perturbation_z = field.get("recent_perturbation_zscore")
    if isinstance(perturbation_z, (int, float)):
        lines.append(f"  perturbation rate z-score: {float(perturbation_z):+.3f}")

    nodes = field.get("node_vectors")
    if isinstance(nodes, dict):
        lines.append(f"  nodes observed: {len(nodes)}")
    edges = field.get("edges")
    if isinstance(edges, list):
        lines.append(f"  edges observed: {len(edges)}")

    if len(lines) == 1:
        lines.append("  (no readings available this tick)")
    return "\n".join(lines)


def build_context(tick: dict[str, Any], verb: str) -> dict[str, Any]:
    """Context in the shape the dispatch runtime sends, plus a real intention.

    Mirrors `orion/execution_dispatch/envelopes.py`'s envelope context keys so
    the model receives what it would receive in production. `intention` is
    added because the prompt template requires it and an empty slot measures
    recall rather than cognition -- see `summarise_field_as_raw_material`.
    """
    field = tick.get("field") or {}
    return {
        "origin": "endogenous.dispatch",
        "target_kind": "system",
        "target_id": f"field:{tick.get('tick_id')}",
        "field_tick_id": tick.get("tick_id"),
        "field_state": field,
        # The template's own slot. Readings only -- no objective supplied.
        "intention": summarise_field_as_raw_material(field),
        "e0_census": True,
        "verb_under_measurement": verb,
    }


# ===========================================================================
# Invocation -- the real bus RPC path
# ===========================================================================


async def run_one(client, verb: str, tick: dict[str, Any], index: int, timeout_sec: float) -> RunResult:
    dispatch_id = f"e0census:{verb}:{tick.get('tick_id')}:{index}"
    started = time.monotonic()
    try:
        payload = await client.dispatch(
            verb=verb,
            mode="brain",
            context=build_context(tick, verb),
            dispatch_id=dispatch_id,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:  # noqa: BLE001 - a failure is a real measurement
        return RunResult(
            verb=verb,
            run_index=index,
            tick_id=tick.get("tick_id"),
            wall_sec=time.monotonic() - started,
            ok=False,
            error=str(exc)[:500],
        )
    wall = time.monotonic() - started

    from orion.execution_dispatch.result_extraction import (
        extract_final_text,
        plan_execution_status,
    )

    plan_status, _ = plan_execution_status(payload)
    final_text = extract_final_text(payload)

    # Token count, best-effort, from wherever cortex-exec surfaced it. Absent
    # is reported as None rather than 0 -- "not measured" and "measured zero"
    # must stay distinguishable.
    tokens: int | None = None
    result = payload.get("result") if isinstance(payload, dict) else None
    if isinstance(result, dict):
        for key in ("completion_tokens", "provider_completion_tokens"):
            value = result.get(key)
            if isinstance(value, int):
                tokens = value
                break
        if tokens is None:
            usage = result.get("usage")
            if isinstance(usage, dict):
                value = usage.get("completion_tokens") or usage.get("total_tokens")
                if isinstance(value, int):
                    tokens = value

    return RunResult(
        verb=verb,
        run_index=index,
        tick_id=tick.get("tick_id"),
        wall_sec=wall,
        ok=True,
        completion_tokens=tokens,
        plan_status=plan_status,
        final_text=final_text,
    )


async def run_census(
    verbs: Sequence[str],
    ticks: Sequence[dict[str, Any]],
    *,
    bus_url: str,
    request_channel: str,
    result_prefix: str,
    timeout_sec: float,
    sleep_sec: float,
) -> list[VerbSummary]:
    from orion.core.bus.async_service import OrionBusAsync
    from orion.execution_dispatch.cortex_client import ExecutionDispatchCortexClient

    # No service_name kwarg on OrionBusAsync -- the census identifies itself
    # through the client's `source_name` below, which is what lands in the
    # envelope and therefore what shows up in cortex-exec's logs.
    bus = OrionBusAsync(url=bus_url)
    await bus.connect()
    client = ExecutionDispatchCortexClient(
        bus,
        request_channel=request_channel,
        result_prefix=result_prefix,
        source_name="e0-cost-census",
        timeout_sec=timeout_sec,
    )

    summaries: list[VerbSummary] = []
    try:
        for verb in verbs:
            summary = VerbSummary(verb=verb)
            for index, tick in enumerate(ticks):
                logger.info("running %s (%d/%d) on %s", verb, index + 1, len(ticks), tick.get("tick_id"))
                result = await run_one(client, verb, tick, index, timeout_sec)
                summary.runs.append(result)
                logger.info(
                    "  -> ok=%s wall=%.2fs tokens=%s status=%s len=%d",
                    result.ok,
                    result.wall_sec,
                    result.completion_tokens,
                    result.plan_status,
                    len(result.final_text),
                )
                # Sequential AND spaced: the resource under test is serialized
                # inference time, and this rides a lane shared with real dispatch.
                await asyncio.sleep(sleep_sec)
            summaries.append(summary)
    finally:
        try:
            await bus.close()
        except Exception:
            pass
    return summaries


# ===========================================================================
# Report
# ===========================================================================


def render_report(summaries: Sequence[VerbSummary]) -> str:
    out: list[str] = []
    add = out.append

    add("=" * 78)
    add("PHASE E0 -- COST CENSUS: what does Orion's expensive cognition cost?")
    add("=" * 78)
    add("")
    add(f"{'verb':44s} {'ok':>4s} {'fail':>5s} {'mean_s':>8s} {'min_s':>7s} {'max_s':>7s} {'tokens':>8s}")

    for summary in summaries:
        ok = len(summary.ok_runs)
        fail = len(summary.runs) - ok
        mean = summary.stat(statistics.mean)
        low = summary.stat(min)
        high = summary.stat(max)
        token_values = [r.completion_tokens for r in summary.ok_runs if r.completion_tokens is not None]
        tokens = f"{statistics.mean(token_values):.0f}" if token_values else "n/a"
        add(
            f"{summary.verb:44s} {ok:4d} {fail:5d} "
            f"{mean if mean is not None else float('nan'):8.2f} "
            f"{low if low is not None else float('nan'):7.2f} "
            f"{high if high is not None else float('nan'):7.2f} {tokens:>8s}"
        )

    add("")
    add("-" * 78)
    add("KILL GATE A -- is serialized inference time actually scarce?")
    add("-" * 78)
    add("")
    add(f"  threshold: expensive cognition must cost >= {KILL_GATE_A_MIN_WALL_SEC:.1f}s wall-clock")
    add(f"  arena tick cadence (measured live): {ARENA_TICK_SEC:.1f}s")
    add("")

    measured = [s for s in summaries if s.ok_runs]
    if not measured:
        add("  VERDICT: UNDETERMINED -- no verb produced a successful run.")
        add("  This is NOT a pass and NOT a fail. Fix invocation, then re-run.")
    else:
        any_expensive = False
        for summary in measured:
            mean = summary.stat(statistics.mean) or 0.0
            ticks_cost = mean / ARENA_TICK_SEC
            verdict = "EXPENSIVE" if mean >= KILL_GATE_A_MIN_WALL_SEC else "cheap"
            if mean >= KILL_GATE_A_MIN_WALL_SEC:
                any_expensive = True
            add(f"  {summary.verb:44s} {mean:7.2f}s  = {ticks_cost:6.1f} arena ticks   {verdict}")
        add("")
        if any_expensive:
            add("  VERDICT: GATE A PASSED. At least one faculty is genuinely expensive.")
            add("  Choosing it really does cost the ticks shown above. Scarcity is physical.")
            add("  -> Plan proceeds to Phase E0's gate B (human reads the goal texts).")
        else:
            add("  VERDICT: GATE A FAILED -- KILL.")
            add("  No faculty costs >= the threshold, so serialized inference time is NOT")
            add("  the scarce resource. The plan's thesis is dead as written.")
            add("  -> Phases 1-5 are CANCELLED pending a different resource. Report and stop.")

    add("")
    add("-" * 78)
    add("KILL GATE B -- is the faculty hollow? (HUMAN JUDGEMENT, not scored here)")
    add("-" * 78)
    add("")
    add("  Deliberately not auto-scored. An LLM grading whether another LLM's output")
    add("  is generic is two untested things agreeing -- banned by the plan's own")
    add("  throttle rules. The verbatim outputs are written to the artifact directory;")
    add("  a human reads the goal_formulate texts and counts how many are things they")
    add("  would not have written. Fewer than 3 of 10 kills Phase 2.")
    add("")
    add("=" * 78)
    return "\n".join(out)


# ===========================================================================
# Entry point
# ===========================================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=10, help="real field ticks per verb")
    parser.add_argument("--verbs", type=str, default=",".join(E0_VERBS))
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=2.0,
        help="pause between calls; this lane is shared with real dispatch",
    )
    parser.add_argument("--out-dir", type=str, default="/tmp/e0-cost-census")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.runs <= 0:
        logger.error("--runs must be positive")
        return 1

    conn = open_readonly_connection(os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI))
    if conn is None:
        return 1
    try:
        ticks = load_real_field_ticks(conn, args.runs)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not ticks:
        logger.error("no real field ticks available; cannot build real context")
        return 1
    logger.info("loaded %d real field ticks", len(ticks))

    verbs = [v.strip() for v in args.verbs.split(",") if v.strip()]
    summaries = asyncio.run(
        run_census(
            verbs,
            ticks,
            bus_url=os.environ.get("ORION_BUS_URL", DEFAULT_BUS_URL),
            request_channel=os.environ.get("CORTEX_EXEC_CHANNEL", DEFAULT_REQUEST_CHANNEL),
            result_prefix=os.environ.get("CORTEX_EXEC_RESULT_PREFIX", DEFAULT_RESULT_PREFIX),
            timeout_sec=args.timeout_sec,
            sleep_sec=args.sleep_sec,
        )
    )

    report = render_report(summaries)
    print(report)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    payload = [
        {
            "verb": s.verb,
            "runs": [
                {
                    "run_index": r.run_index,
                    "tick_id": r.tick_id,
                    "wall_sec": round(r.wall_sec, 3),
                    "ok": r.ok,
                    "error": r.error,
                    "completion_tokens": r.completion_tokens,
                    "plan_status": r.plan_status,
                    "final_text": r.final_text,
                }
                for r in s.runs
            ],
        }
        for s in summaries
    ]
    with open(os.path.join(args.out_dir, "runs.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    # The material for gate B, in the form a human actually reads.
    with open(os.path.join(args.out_dir, "outputs.md"), "w", encoding="utf-8") as handle:
        for s in summaries:
            handle.write(f"\n\n# {s.verb}\n")
            for r in s.runs:
                handle.write(f"\n## run {r.run_index} (tick {r.tick_id}, {r.wall_sec:.2f}s)\n\n")
                handle.write((r.final_text or "(no output)") + "\n")
    logger.info("artifacts written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
