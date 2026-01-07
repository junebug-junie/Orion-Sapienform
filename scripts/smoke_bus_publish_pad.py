#!/usr/bin/env python3
"""
Smoke test: publish a Titanium envelope into the Landing Pad intake and
verify pad.event.v1 + pad.frame.v1 are emitted.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


BUS_URL = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
INPUT_CHANNEL = os.getenv("PAD_SMOKE_INPUT_CHANNEL", "orion:telemetry:smoke")
EVENT_CHANNEL = os.getenv("PAD_OUTPUT_EVENT_CHANNEL", "orion:pad:event")
FRAME_CHANNEL = os.getenv("PAD_OUTPUT_FRAME_CHANNEL", "orion:pad:frame")
TIMEOUT_SEC = float(os.getenv("PAD_SMOKE_TIMEOUT_SEC", "20"))


async def main() -> int:
    bus = OrionBusAsync(BUS_URL)
    await bus.connect()

    envelope = BaseEnvelope(
        kind="telemetry.smoke.v1",
        source=ServiceRef(name="smoke-landing-pad", node="local", version="0.0.0"),
        correlation_id=uuid4(),
        payload={
            "metric": "smoke-test",
            "value": 1.0,
            "ts_ms": int(time.time() * 1000),
        },
    )

    seen = set()
    async with bus.subscribe(EVENT_CHANNEL, FRAME_CHANNEL) as pubsub:
        await bus.publish(INPUT_CHANNEL, envelope)

        start = time.time()
        async for msg in bus.iter_messages(pubsub):
            if time.time() - start > TIMEOUT_SEC:
                raise TimeoutError("Timed out waiting for pad outputs")

            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok or not decoded.envelope:
                continue

            kind = decoded.envelope.kind
            if kind == "orion.pad.event.v1":
                seen.add("event")
                print("[ok] received pad.event.v1")
            elif kind == "orion.pad.frame.v1":
                seen.add("frame")
                print("[ok] received pad.frame.v1")

            if {"event", "frame"}.issubset(seen):
                break

    await bus.close()
    if {"event", "frame"}.issubset(seen):
        print("Landing Pad smoke passed.")
        return 0
    print("Landing Pad smoke failed.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
