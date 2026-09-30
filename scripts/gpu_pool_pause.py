#!/usr/bin/env python3
"""GPU pool emergency stop from a shell: pause (or resume) every model load and unload.

The same control verb the Hub GPU pool panel's "Emergency stop" button sends (stage 5.7). Use it when the
Hub is down. Persisted by the pool on gpu_pool_cards, so it stays paused across a pool restart until resumed.
An action already running on circe is NOT stopped by this (stop orion-circe-gpu-lane-controller for that).
Runbook: docs/runbooks/2026-09-30-gpu-pool-stage5-7-enforce.md.

    ORION_BUS_URL=redis://100.92.216.81:6379/0 PYTHONPATH=. .venv/bin/python scripts/gpu_pool_pause.py pause
    ... scripts/gpu_pool_pause.py resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.gpu_pool import (  # noqa: E402
    GPU_POOL_CONTROL_KIND, GPU_POOL_CONTROL_REPLY_PREFIX, GPU_POOL_CONTROL_REQUEST_CHANNEL, GpuPoolControlReplyV1,
    GpuPoolControlV1,
)

VERBS = {"pause": "pause_actuation", "resume": "resume_actuation"}
SOURCE = "operator:gpu-pool-pause"


def build(action: str, actor: str) -> tuple[str, BaseEnvelope]:
    """(reply channel, request envelope) for ``pause`` / ``resume``."""
    reply = f"{GPU_POOL_CONTROL_REPLY_PREFIX}{uuid.uuid4().hex}"
    ctl = GpuPoolControlV1(verb=VERBS[action], actor=actor)
    return reply, BaseEnvelope(kind=GPU_POOL_CONTROL_KIND, source=ServiceRef(name=SOURCE), correlation_id=uuid.uuid4(),
                               reply_to=reply, payload=ctl.model_dump(mode="json"))


def parse(raw: Any) -> GpuPoolControlReplyV1:
    """The pool's reply out of the raw pub/sub message ``rpc_request`` returns (or an envelope dict)."""
    data = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
    env = json.loads(data) if isinstance(data, (str, bytes, bytearray)) else data
    return GpuPoolControlReplyV1.model_validate(env.get("payload") if isinstance(env, dict) else None)


def describe(action: str, reply: GpuPoolControlReplyV1) -> str:
    if not reply.ok:
        return f"REFUSED: {reply.reason}"
    d = reply.detail
    if d.get("paused"):
        running = d.get("in_flight") or []
        tail = f"; still running on circe (not stopped): {running}" if running else ""
        return f"PAUSED since {d.get('since')} by {d.get('by')} ({reply.reason}){tail}"
    return f"RUNNING: model loading/unloading is on ({reply.reason})"


async def send(action: str, actor: str, timeout_sec: float) -> int:
    from orion.core.bus.async_service import OrionBusAsync

    bus = OrionBusAsync(url=os.environ["ORION_BUS_URL"])
    await bus.connect()
    try:
        reply_channel, env = build(action, actor)
        raw = await bus.rpc_request(GPU_POOL_CONTROL_REQUEST_CHANNEL, env, reply_channel=reply_channel,
                                    timeout_sec=timeout_sec)
        reply = parse(raw)
    except asyncio.TimeoutError:
        print("NO ANSWER: the pool did not reply (down, or not the leader yet). Nothing changed.")
        return 2
    finally:
        await bus.close()
    print(describe(action, reply))
    return 0 if reply.ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=sorted(VERBS))
    ap.add_argument("--actor", default=os.environ.get("USER") or "operator-shell")
    ap.add_argument("--timeout-sec", type=float, default=15.0)
    args = ap.parse_args()
    return asyncio.run(send(args.action, args.actor, args.timeout_sec))


if __name__ == "__main__":
    sys.exit(main())
