#!/usr/bin/env python3
"""Patch 1 of docs/superpowers/specs/2026-09-09-curiosity-supervisor-design.md.

Reads every `Hop` Orion has ever written to `orion_worldview`, asks the
cortex brain lane to produce a `HopReadingV1` per hop, and writes them out for
a human to sample. Nothing here writes to FalkorDB or Postgres, and nothing
publishes an intervention -- the sole write is two local files under
`--out-dir`.

    python3 scripts/report_curiosity_supervisor_readings.py
    python3 scripts/report_curiosity_supervisor_readings.py --json

Needs FalkorDB (HUB_CURIOSITY_GRAPH_HOST/PORT/OWN, localhost by default) and
ORION_BUS_URL (no default here -- this repo's standing rule is that value is
never guessed; set it to the real tailscale bus address or pass --bus-url).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.core.bus.async_service import OrionBusAsync  # noqa: E402
from orion.core.bus.bus_schemas import ServiceRef  # noqa: E402
from orion.curiosity.supervisor import (  # noqa: E402
    DEFAULT_LLM_ROUTE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_SEC,
    generate_all_readings,
    group_readings_by_prior,
    is_circling,
)
from orion.curiosity.worldview import WorldviewReader, read_all_hops, read_all_priors  # noqa: E402


def _write_outputs(
    out_dir: Path, readings, priors_by_id: dict[str, str], *, total_hops_read: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "readings.jsonl").open("w") as fh:
        for r in readings:
            fh.write(json.dumps(r.model_dump(mode="json")) + "\n")

    by_run = _group_by_run(readings)
    by_prior = group_readings_by_prior(readings)
    lines = [
        "# Curiosity supervisor -- hop readings (Patch 1)",
        "",
        # `total_hops_read` (from read_all_hops) vs `len(readings)` are the
        # two numbers that diverge when something goes wrong -- a truncated
        # or malformed LLM response drops a whole run's readings silently
        # (see parse_reading_batch), so reporting only the produced count
        # under a "hops read" label would hide exactly that failure from the
        # person this report is for. Caught in review.
        f"Hops read from the graph: {total_hops_read}. Readings produced: "
        f"{len(readings)}, across {len(by_run)} runs.",
        f"Attributed to a prior: {sum(len(v) for v in by_prior.values())} / {len(readings)}.",
        "",
        "## Per-prior circling verdicts",
        "",
        "`is_circling`: None = not enough evidence (<3 attributed hops), "
        "True = last 3 attributed hops all tested the claim and moved "
        "nothing, False = otherwise.",
        "",
    ]
    for prior_id, prior_readings in sorted(by_prior.items(), key=lambda kv: -len(kv[1])):
        verdict = is_circling(prior_readings)
        claim = priors_by_id.get(prior_id, "(unknown prior -- not in current live+closed set)")
        lines.append(f"- `{prior_id}` ({len(prior_readings)} hops, circling={verdict}): {claim}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def _group_by_run(readings) -> dict[str, list]:
    out: dict[str, list] = {}
    for r in readings:
        out.setdefault(r.hop_run_id, []).append(r)
    return out


async def _run(args: argparse.Namespace) -> int:
    reader = WorldviewReader(host=args.graph_host, port=args.graph_port, graph_name=args.graph_name)
    priors_by_id = {p.prior_id: p.claim for p in read_all_priors(reader)}
    # Read once, here, purely for the report's own headline count --
    # generate_all_readings does its own (identical) read internally. A
    # second cheap read-only FalkorDB query is a fair price for the report
    # being able to say how many hops it STARTED from, not just how many it
    # finished with.
    total_hops_read = len(read_all_hops(reader))

    bus = OrionBusAsync(url=args.bus_url)
    await bus.connect()
    try:
        source = ServiceRef(name="curiosity-supervisor-readings", node="athena", version="0.0.1")

        def _progress(run_id: str, readings) -> None:
            print(f"  run {run_id}: {len(readings)} reading(s)", file=sys.stderr)

        readings = await generate_all_readings(
            bus,
            reader,
            cortex_request_channel=args.cortex_request_channel,
            cortex_result_prefix=args.cortex_result_prefix,
            source=source,
            llm_route=args.llm_route,
            timeout_sec=args.timeout_sec,
            max_tokens=args.max_tokens,
            on_run_done=_progress,
        )
    finally:
        await bus.close()

    out_dir = Path(args.out_dir)
    _write_outputs(out_dir, readings, priors_by_id, total_hops_read=total_hops_read)

    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in readings], default=str))
        return 0

    by_prior = group_readings_by_prior(readings)
    print()
    print(
        f"=== {total_hops_read} hops read, {len(readings)} readings produced, "
        f"{len(_group_by_run(readings))} runs ==="
    )
    print(f"  written: {out_dir}/readings.jsonl, {out_dir}/report.md")
    print()
    print("=== Per-prior circling verdicts ===")
    if not by_prior:
        print("  no readings attributed to any prior")
    for prior_id, prior_readings in sorted(by_prior.items(), key=lambda kv: -len(kv[1])):
        verdict = is_circling(prior_readings)
        claim = priors_by_id.get(prior_id, "(unknown prior)")
        claim = claim if len(claim) <= 80 else claim[:77] + "..."
        print(f"  {len(prior_readings):>3} hops  circling={str(verdict):<5}  {prior_id}: {claim}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-host", default=os.environ.get("HUB_CURIOSITY_GRAPH_HOST", "127.0.0.1"))
    ap.add_argument("--graph-port", type=int, default=int(os.environ.get("HUB_CURIOSITY_GRAPH_PORT", "6380")))
    ap.add_argument("--graph-name", default=os.environ.get("HUB_CURIOSITY_GRAPH_OWN", "orion_worldview"))
    ap.add_argument("--bus-url", default=os.environ.get("ORION_BUS_URL"),
                     help="No default guessed here -- set ORION_BUS_URL or pass this explicitly.")
    ap.add_argument("--cortex-request-channel",
                     default=os.environ.get("CORTEX_REQUEST_CHANNEL", "orion:cortex:request"))
    ap.add_argument("--cortex-result-prefix",
                     default=os.environ.get("CORTEX_RESULT_PREFIX", "orion:cortex:result"))
    ap.add_argument("--llm-route", default=DEFAULT_LLM_ROUTE)
    ap.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--out-dir", default="/tmp/curiosity-supervisor-hop-readings")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.bus_url:
        print("ERROR: ORION_BUS_URL is not set and --bus-url was not passed. "
              "This repo's rule is that value is never guessed -- see CLAUDE.md.",
              file=sys.stderr)
        return 2
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
