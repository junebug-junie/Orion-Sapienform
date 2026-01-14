#!/usr/bin/env python3
"""
Smoke test: verify deep.graph.v1 recall profile reaches RecallService via Cortex-Orch.

Usage:
    python scripts/recall_smoke_deep_graph.py --redis redis://localhost:6379/0
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from uuid import uuid4
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef, LLMMessage
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, RecallDirective

ORCH_CHANNEL = os.getenv("CORTEX_ORCH_REQUEST_CHANNEL", "orion-cortex:request")
ORCH_RESULT_PREFIX = os.getenv("CORTEX_ORCH_RESULT_PREFIX", "orion-cortex:result")


def _source() -> ServiceRef:
    return ServiceRef(name="recall-smoke-deep-graph", node="local", version="0.0.1")


async def _rpc(bus: OrionBusAsync, payload: CortexClientRequest, *, timeout_sec: float = 120.0) -> dict:
    corr = str(uuid4())
    reply_channel = f"{ORCH_RESULT_PREFIX}:{corr}"
    env = BaseEnvelope(
        kind="cortex.orch.request",
        source=_source(),
        correlation_id=corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(ORCH_CHANNEL, env, reply_channel=reply_channel, timeout_sec=timeout_sec)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error)
    return decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else decoded.envelope.payload.model_dump(mode="json")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test deep.graph.v1 recall profile.")
    parser.add_argument("--redis", type=str, default=os.getenv("ORION_BUS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--query", type=str, default="What do we know about Orion recall graphs?")
    args = parser.parse_args()

    bus = OrionBusAsync(url=args.redis)
    await bus.connect()

    ctx = CortexClientContext(
        messages=[LLMMessage(role="user", content=args.query)],
        session_id="smoke-deep-graph",
    )
    payload = CortexClientRequest(
        mode="brain",
        verb_name="chat_deep_graph",
        packs=["executive_pack"],
        recall=RecallDirective(enabled=True),
        context=ctx,
    )

    try:
        res = await _rpc(bus, payload, timeout_sec=args.timeout)
    finally:
        await bus.close()

    recall_debug = res.get("recall_debug") if isinstance(res, dict) else {}
    profile = recall_debug.get("profile") if isinstance(recall_debug, dict) else None

    if profile != "deep.graph.v1":
        print("Smoke test failed: expected recall profile deep.graph.v1, got:", profile)
        return 1

    print("Smoke test passed: Recall profile deep.graph.v1 received.")
    print("Recall debug:", recall_debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
