#!/usr/bin/env python3
"""Keep Orion's own work off circe gpu2 (both Bonsai slots) while scripts/gpu_power_bench.py runs.

gpu2's agent lane is never quiet for long: during a 40 min two-slot bench (2026-10-29) cortex-exec
put a call on gpu2 every one to two minutes, so every round was thrown away as dirty. This takes
operator holds through the GPU pool, prints READY, and keeps them until Ctrl+C, SIGTERM/SIGHUP, or
--minutes run out; every exit path releases them (orion.evals.model_replay.pool_hold.PoolHolds):

  1. gpu1 (role agent, class memory_distill): its only slot. Without it, class `agent` lands there
     first (the class's route is agent -> agent-gpu2 -> agent-deep -> chat).
  2. gpu2 (role agent-gpu2, class agent): both slots (max_holds 2). No class routes to gpu2 alone,
     so the second hold goes out only when the pool shows gpu2's other slot free; otherwise it falls
     through to the next free card (first live try, 2026-10-29: the chat card, given back in 0.15 s).
     A hold that still lands elsewhere is given back at once and retried (--extra-tries); if gpu2's
     second slot never comes, everything is released: half of gpu2 is not a clean card.

Cost: the agent lane has no circe card while this runs; its work waits in the pool queue. Operator
holds are priority interactive, so only work of strictly higher priority may use their gaps (H3).

    ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python scripts/gpu_pool_clear_gpu2.py
    ... --release-leftovers OUT_DIR   # after a SIGKILL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from orion.evals.model_replay.pool_hold import (
    SEATS, BusControlTransport, HoldLedger, HoldRefused, PoolHolds, http_sweeper, release_leftovers,
)

ACTOR = "gpu-power-bench"
DEFAULT_POOL_HTTP = "http://localhost:8127"
HOLDER = f"operator:{ACTOR}"
GPU2_ROLE = SEATS["bonsai"].expect_role
LIVE = ("granted", "recalling")


def gpu2_has_free_slot(pool_state: dict[str, Any], holder: str = HOLDER) -> bool:
    """True when nothing but our own holds is live on gpu2's role, so a class-agent hold lands there."""
    return not any(lease.get("role") == GPU2_ROLE and lease.get("status") in LIVE
                   and lease.get("holder") != holder for lease in pool_state.get("leases") or [])


def http_pool_state(pool_http: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{pool_http}/v1/pool", timeout=10) as resp:
        return json.loads(resp.read())


async def take_second_gpu2_slot(holds: PoolHolds, pool_state, tries: int, wait_sec: float) -> bool:
    """One more class-agent hold, sent only while gpu2's other slot is free; True once it is ours.
    A hold that lands elsewhere is given back at once by _acquire_extra (it checks the role)."""
    deadline = time.monotonic() + wait_sec
    for _ in range(tries):
        while True:
            try:
                if gpu2_has_free_slot(await asyncio.to_thread(pool_state)):
                    break
            except Exception:  # noqa: BLE001 -- pool HTTP hiccup: keep waiting until the deadline
                pass
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.5)
        before = len(holds._held)
        await holds._acquire_extra(SEATS["bonsai"])
        if len(holds._held) > before:
            return True
        print("  second gpu2 hold landed elsewhere and was given back; retrying", flush=True)
        await asyncio.sleep(1.0)
    return False


async def hold(args: argparse.Namespace, bus_url: str) -> int:
    ledger = HoldLedger(args.out / "holds.jsonl")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, stop.set)
    async with BusControlTransport(bus_url) as transport:
        holds = PoolHolds(transport=transport, ledger=ledger, actor=ACTOR,
                          grant_timeout_sec=args.grant_timeout_sec, sweep=http_sweeper(args.pool_http, HOLDER))
        print(f"asking the pool for gpu1 and both gpu2 slots (waits up to {args.grant_timeout_sec / 60:.0f} min "
              f"for running work to finish); ledger {ledger.path}", flush=True)
        acquire = asyncio.ensure_future(holds.__aenter__())
        stopped = asyncio.ensure_future(stop.wait())
        await asyncio.wait([acquire, stopped], return_when=asyncio.FIRST_COMPLETED)
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
            second = asyncio.ensure_future(take_second_gpu2_slot(
                holds, lambda: http_pool_state(args.pool_http), args.extra_tries, args.grant_timeout_sec))
            await asyncio.wait([second, stopped], return_when=asyncio.FIRST_COMPLETED)
            if not second.done():
                second.cancel()
                await asyncio.gather(second, return_exceptions=True)
                print("stopped while waiting for gpu2's second slot", flush=True)
                return 130
            # 3 = gpu1 + gpu2 + gpu2's other slot; with only one gpu2 slot ours the card is not clean.
            if not second.result() or len(holds._held) != 3:
                print(f"got {len(holds._held)} of 3 holds (gpu2's second slot never came free in "
                      f"{args.extra_tries} tries); releasing everything", file=sys.stderr)
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
                                       sweep=http_sweeper(args.pool_http, HOLDER))
    print(f"released {len(left)} leftover hold(s): {left}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--minutes", type=float, default=60.0, help="release after this long at the latest")
    # gpu2 may be unloaded for image generation: the pool reloads it within ~20 min of agent demand.
    ap.add_argument("--grant-timeout-sec", type=float, default=1800.0,
                    help="per seat hold, and for gpu2's second slot to come free")
    ap.add_argument("--pool-http", default=DEFAULT_POOL_HTTP)
    ap.add_argument("--extra-tries", type=int, default=5,
                    help="tries for gpu2's second slot; each miss holds another card for well under a second")
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
