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

from .settings import settings
from .worker import handle_candidate

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
    svc = Hunter(
        _cfg(),
        patterns=[settings.channel_spark_candidate],
        handler=handle_candidate,
    )
    logger.info("Starting Spark Introspector Hunter patterns=%s", [settings.channel_spark_candidate])
    await svc.start()


if __name__ == "__main__":
    asyncio.run(main())
