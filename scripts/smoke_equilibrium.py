#!/usr/bin/env python3
"""
Smoke test: publish system.health.v1 heartbeats and confirm the equilibrium
service emits equilibrium.snapshot.v1 and spark.signal.v1.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1


BUS_URL = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
HEALTH_CHANNEL = os.getenv("CHANNEL_SYSTEM_HEALTH", "system.health")
SNAPSHOT_CHANNEL = os.getenv("CHANNEL_EQUILIBRIUM_SNAPSHOT", "orion:equilibrium:snapshot")
SPARK_SIGNAL_CHANNEL = os.getenv("CHANNEL_SPARK_SIGNAL", "orion:spark:signal")
TIMEOUT_SEC = float(os.getenv("EQUILIBRIUM_SMOKE_TIMEOUT_SEC", "30"))


async def publish_heartbeats(bus: OrionBusAsync) -> None:
    now = datetime.now(timezone.utc)
    services = ("smoke-svc-a", "smoke-svc-b")
    for svc in services:
        hb = SystemHealthV1(
            service=svc,
            node="local",
            version="0.0.0",
            instance=str(uuid4()),
            boot_id=str(uuid4()),
            status="ok",
            last_seen_ts=now,
            heartbeat_interval_sec=5.0,
            details={"smoke": True},
        )
        env = BaseEnvelope(kind="system.health.v1", source=ServiceRef(name=svc, node="local", version="0.0.0"), payload=hb.model_dump(mode="json"))
        await bus.publish(HEALTH_CHANNEL, env)


async def main() -> int:
    bus = OrionBusAsync(BUS_URL)
    await bus.connect()
    await publish_heartbeats(bus)

    seen = set()
    start = time.time()
    async with bus.subscribe(SNAPSHOT_CHANNEL, SPARK_SIGNAL_CHANNEL) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            if time.time() - start > TIMEOUT_SEC:
                raise TimeoutError("Timed out waiting for equilibrium outputs")

            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok or not decoded.envelope:
                continue

            kind = decoded.envelope.kind
            if kind == "equilibrium.snapshot.v1":
                seen.add("snapshot")
                print("[ok] received equilibrium.snapshot.v1")
            elif kind == "spark.signal.v1":
                seen.add("signal")
                print("[ok] received spark.signal.v1")

            if {"snapshot", "signal"}.issubset(seen):
                break

    await bus.close()
    if {"snapshot", "signal"}.issubset(seen):
        print("Equilibrium smoke passed.")
        return 0
    print("Equilibrium smoke failed.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
