#!/usr/bin/env python3
"""Pool controls for the stage 7.1 bake-off: pause/resume actuation, hold/release diffusion.

Runs where the repo's Python deps live. circe has no venv, so stage7_1_bakeoff.sh runs it inside
the Bonsai image (it already carries redis + pydantic for the worker announce), with the repo
mounted and on sys.path first:

    docker run --rm --network host -v "$ROOT:/repo:ro" -e ORION_BUS_URL=... --entrypoint python3 \
        llamacpp-bonsai-prism:server-local-volta /repo/scripts/bench/stage7_1_pool_ctl.py pause

Same control verbs the Hub panel and scripts/gpu_pool_pause.py send (GpuPoolControlV1).

Why hold diffusion: diffusion-host stays up on gpu2 and loads its ~24 GB model on demand (a grant
lands every ~1-2 h, 2026-09-28..10-01). Pausing actuation does not stop that. An operator hold on
the diffusion role makes that work wait in the pool queue instead of running out of memory next
to Bonsai, and world.serialize_with keeps world-model compute off the card meanwhile. An operator
hold has no heartbeat and diffusion has no max_hold_sec: it lasts until released, so the bake-off
script releases it on every exit path and `cleanup` releases any live hold with this holder.

Prints one JSON line: {"ok": ..., "reason": ..., "detail": {...}}. Exit 0 only when ok.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.gpu_pool import (  # noqa: E402
    GPU_POOL_CONTROL_KIND, GPU_POOL_CONTROL_REPLY_PREFIX, GPU_POOL_CONTROL_REQUEST_CHANNEL, GpuPoolControlV1,
)
from scripts.gpu_pool_pause import parse  # noqa: E402  -- the one reply parser, reused

ACTOR = "stage7-1-bakeoff"
SOURCE = "operator:stage7-1-bakeoff"
HOLD_CLASS = "diffusion"
VERBS = {"pause": "pause_actuation", "resume": "resume_actuation", "hold": "hold", "release": "release"}


def build(action: str, actor: str = ACTOR, lease_id: str | None = None) -> tuple[str, BaseEnvelope]:
    """(reply channel, request envelope) for one control verb."""
    if action == "release" and not lease_id:
        raise ValueError("release needs --lease-id")
    ctl = GpuPoolControlV1(verb=VERBS[action], actor=actor,
                           work_class=HOLD_CLASS if action == "hold" else None,
                           lease_id=lease_id if action == "release" else None)
    reply = f"{GPU_POOL_CONTROL_REPLY_PREFIX}{uuid.uuid4().hex}"
    return reply, BaseEnvelope(kind=GPU_POOL_CONTROL_KIND, source=ServiceRef(name=SOURCE),
                               correlation_id=uuid.uuid4(), reply_to=reply, payload=ctl.model_dump(mode="json"))


async def send(action: str, actor: str, lease_id: str | None, timeout_sec: float) -> int:
    from orion.core.bus.async_service import OrionBusAsync

    reply_channel, env = build(action, actor, lease_id)
    bus = OrionBusAsync(url=os.environ["ORION_BUS_URL"])
    await bus.connect()
    try:
        raw = await bus.rpc_request(GPU_POOL_CONTROL_REQUEST_CHANNEL, env, reply_channel=reply_channel,
                                    timeout_sec=timeout_sec)
        reply = parse(raw)
    except asyncio.TimeoutError:
        print(json.dumps({"ok": False, "reason": "no_answer", "detail": {}}))
        return 2
    finally:
        await bus.close()
    print(json.dumps(reply.model_dump(mode="json"), default=str))
    return 0 if reply.ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=sorted(VERBS) + ["check"])
    ap.add_argument("--actor", default=ACTOR)
    ap.add_argument("--lease-id")
    ap.add_argument("--timeout-sec", type=float, default=15.0)
    args = ap.parse_args()
    if args.action == "check":   # imports resolved + bus reachable; sends nothing to the pool
        import redis

        from orion.core.bus.async_service import OrionBusAsync  # noqa: F401 -- what the real verbs use
        redis.Redis.from_url(os.environ["ORION_BUS_URL"], socket_timeout=5).ping()
        print(json.dumps({"ok": True, "reason": "imports_and_bus_ok", "detail": {}}))
        return 0
    return asyncio.run(send(args.action, args.actor, args.lease_id, args.timeout_sec))


if __name__ == "__main__":
    sys.exit(main())
