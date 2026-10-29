#!/usr/bin/env python3
"""Keep Orion's own work off circe gpu2 (both Bonsai slots) while scripts/gpu_power_bench.py runs.

gpu2's agent lane is never quiet for long: during a 40 min two-slot bench (2026-10-29) cortex-exec
put a call on gpu2 every one to two minutes, so every round was thrown away as dirty. This takes
operator holds through the GPU pool, prints READY, and keeps them until Ctrl+C, SIGTERM/SIGHUP, or
--minutes run out; every exit path releases them (orion.evals.model_replay.pool_hold.PoolHolds):

  1. gpu1 (role agent, class memory_distill): its only slot. Without it, class `agent` lands there
     first (the class's route is agent -> agent-gpu2 -> agent-deep -> chat).
  2. gpu2 (role agent-gpu2, class agent): both slots (max_holds 2). A hold that lands anywhere else
     (hecate, the chat card) is given back at once, and then everything is released: half of gpu2
     is not a clean card.

Cost: the agent lane has no circe card while this runs; its work waits in the pool queue. Operator
holds are priority interactive, so only work of strictly higher priority may use their gaps (H3).

    ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python scripts/gpu_pool_clear_gpu2.py
    ... --release-leftovers OUT_DIR   # after a SIGKILL
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

from orion.evals.model_replay.pool_hold import (
    SEATS, BusControlTransport, HoldLedger, HoldRefused, PoolHolds, http_sweeper, release_leftovers,
)

ACTOR = "gpu-power-bench"
DEFAULT_POOL_HTTP = "http://localhost:8127"
# Both gpu2 slots: the replay's Bonsai seat plus its other slot.
SPECS = {"q4": SEATS["q4"], "bonsai": replace(SEATS["bonsai"], extra_slot_holds=1)}


async def hold(args: argparse.Namespace, bus_url: str) -> int:
    ledger = HoldLedger(args.out / "holds.jsonl")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, stop.set)
    async with BusControlTransport(bus_url) as transport:
        holds = PoolHolds(transport=transport, ledger=ledger, actor=ACTOR, specs=SPECS,
                          grant_timeout_sec=args.grant_timeout_sec, extra_grant_timeout_sec=args.grant_timeout_sec,
                          sweep=http_sweeper(args.pool_http, f"operator:{ACTOR}"))
        print(f"asking the pool for gpu1 and both gpu2 slots (waits up to {args.grant_timeout_sec / 60:.0f} min "
              f"for running work to finish); ledger {ledger.path}", flush=True)
        acquire = asyncio.ensure_future(holds.__aenter__())
        await asyncio.wait([acquire, asyncio.ensure_future(stop.wait())], return_when=asyncio.FIRST_COMPLETED)
        if not acquire.done():
            acquire.cancel()  # __aenter__ releases what it took on any exception, cancellation included
            await asyncio.gather(acquire, return_exceptions=True)
            print("stopped before the holds were complete; everything released", flush=True)
            return 130
        try:
            seats = acquire.result()
        except HoldRefused as exc:
            print(f"pool refused: {exc}; everything released", file=sys.stderr)
            return 2
        try:
            # 3 = gpu1 + gpu2 + gpu2's other slot. _acquire_extra already gave back a hold that
            # landed elsewhere; with only one gpu2 slot ours the card is not clean.
            if len(holds._held) != 3:
                print(f"got {len(holds._held)} of 3 holds (gpu2's second slot went elsewhere or timed out); "
                      f"releasing everything", file=sys.stderr)
                return 2
            print(f"READY: gpu1 + both gpu2 slots held ({', '.join(f'{k}={g.role}' for k, g in seats.items())}). "
                  f"Run the bench now. Releasing in {args.minutes:.0f} min or on Ctrl+C.", flush=True)
            started = time.monotonic()
            try:
                await asyncio.wait_for(stop.wait(), timeout=args.minutes * 60)
            except asyncio.TimeoutError:
                pass
            print(f"releasing after {(time.monotonic() - started) / 60:.1f} min", flush=True)
            return 0
        finally:
            await holds.release_all(reason="gpu_power_bench_done")
            print("released", flush=True)


async def leftovers(args: argparse.Namespace, bus_url: str) -> int:
    async with BusControlTransport(bus_url) as transport:
        left = await release_leftovers(transport, HoldLedger(args.release_leftovers / "holds.jsonl"), ACTOR,
                                       sweep=http_sweeper(args.pool_http, f"operator:{ACTOR}"))
    print(f"released {len(left)} leftover hold(s): {left}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--minutes", type=float, default=60.0, help="release after this long at the latest")
    # gpu2 may be unloaded for image generation: the pool reloads it within ~20 min of agent demand.
    ap.add_argument("--grant-timeout-sec", type=float, default=1800.0, help="per hold")
    ap.add_argument("--pool-http", default=DEFAULT_POOL_HTTP)
    ap.add_argument("--out", type=Path, default=Path.home() / ".orion" / "gpu-power-bench-holds"
                    / time.strftime("%Y%m%dT%H%M%S"))
    ap.add_argument("--release-leftovers", type=Path, metavar="OUT_DIR",
                    help="release holds a killed run left behind (reads OUT_DIR/holds.jsonl and the pool)")
    args = ap.parse_args(argv)
    bus_url = os.environ.get("ORION_BUS_URL")
    if not bus_url:
        print("ORION_BUS_URL is not set (use redis://100.92.216.81:6379/0)", file=sys.stderr)
        return 2
    return asyncio.run(leftovers(args, bus_url) if args.release_leftovers else hold(args, bus_url))


if __name__ == "__main__":
    sys.exit(main())
