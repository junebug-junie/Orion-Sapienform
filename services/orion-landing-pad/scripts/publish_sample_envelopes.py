#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from uuid import uuid4

from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


BUS_URL = os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")
CHANNEL_METRIC = os.getenv("PAD_SAMPLE_METRIC_CHANNEL", "orion:telemetry:metrics")
CHANNEL_SNAPSHOT = os.getenv("PAD_SAMPLE_SNAPSHOT_CHANNEL", "orion:telemetry:snapshots")
CHANNEL_UNKNOWN = os.getenv("PAD_SAMPLE_UNKNOWN_CHANNEL", "orion:telemetry:unknown")


async def publish_samples() -> None:
    bus = OrionBusAsync(BUS_URL)
    await bus.connect()

    source = ServiceRef(name="pad-sample-publisher", node="local", version="0.0.0")

    metric_env = BaseEnvelope(
        kind="telemetry.metric.v1",
        source=source,
        payload={
            "metric": "cpu.utilization",
            "value": 0.42,
            "salience": 0.3,
            "confidence": 0.6,
        },
    )
    snapshot_env = BaseEnvelope(
        kind="spark.state.snapshot.v1",
        source=source,
        payload={
            "source_node": "athena",
            "seq": 1,
            "snapshot_ts": "2024-01-01T00:00:00Z",
            "summary": {"tasks": 5},
        },
    )
    unknown_env = BaseEnvelope(
        kind="unknown.kind.v1",
        source=source,
        payload={"note": "should hit fallback", "id": str(uuid4())},
    )

    logger.info("Publishing sample envelopes...")
    await bus.publish(CHANNEL_METRIC, metric_env)
    await bus.publish(CHANNEL_SNAPSHOT, snapshot_env)
    await bus.publish(CHANNEL_UNKNOWN, unknown_env)
    logger.info("Done.")
    await bus.close()


if __name__ == "__main__":
    asyncio.run(publish_samples())
