"""Spark Introspector

Consumes Spark introspection candidates and asks Cortex-Orch to run an
introspection step (LLM-backed), then re-publishes a completed candidate
payload for downstream persistence (e.g. SQL writer).

This service uses the Orion V2 chassis (Hunter + RPC client).
"""

from __future__ import annotations

import asyncio
import logging

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.codec import OrionCodec
from .settings import settings
from .worker import handle_candidate, handle_trace, set_publisher_bus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orion-spark-introspector")


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=settings.node_name,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=settings.heartbeat_interval_sec,
    )


async def main() -> None:
    # Initialize shared publisher bus
    pub_bus = OrionBusAsync(settings.orion_bus_url, enabled=settings.orion_bus_enabled, codec=OrionCodec())
    await pub_bus.connect()

    # Pass bus to worker
    set_publisher_bus(pub_bus)

    async def multiplexer(env):
        if env.kind == "cognition.trace":
            await handle_trace(env)
        else:
            await handle_candidate(env)

    patterns = [settings.channel_spark_candidate, settings.channel_cognition_trace_pub]

    svc = Hunter(
        _cfg(),
        patterns=patterns,
        handler=multiplexer,
    )
    logger.info("Starting Spark Introspector Hunter patterns=%s", patterns)

    try:
        await svc.start()
    finally:
        await pub_bus.close()


if __name__ == "__main__":
    asyncio.run(main())
