# FILE: services/orion-agent-chain/app/main.py
from __future__ import annotations

import logging
import asyncio

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1

from .settings import settings
from .bus_listener import start_agent_chain_bus_listener

logging.basicConfig(
    level=logging.INFO,
    format="[AGENT-CHAIN] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("agent-chain.main")

app = FastAPI(
    title="Orion Agent Chain",
    version=settings.service_version,
)


heartbeat_task = None

async def heartbeat_loop(settings):
    """Publishes a heartbeat every 30 seconds."""
    # Initialize a local bus just for this loop
    bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
    if bus.enabled:
        await bus.connect()

    logger.info("Heartbeat loop started.")
    try:
        while True:
            if bus.enabled:
                try:
                    payload = SystemHealthV1(
                        service=settings.service_name,
                        version=settings.service_version,
                        node="agent-chain-node",
                        status="ok"
                    ).model_dump(mode="json")

                    await bus.publish("orion:system:health", BaseEnvelope(
                        kind="system.health.v1",
                        source=ServiceRef(name=settings.service_name, version=settings.service_version),
                        payload=payload
                    ))
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")

            await asyncio.sleep(30)
    except asyncio.CancelledError:
        logger.info("Heartbeat loop stopping...")
    finally:
        if bus.enabled:
            await bus.close()

@app.on_event("startup")
def on_startup() -> None:
    global heartbeat_task
    logger.info(
        f"Starting Agent Chain (service={settings.service_name} "
        f"v={settings.service_version}, port={settings.port})"
    )
    start_agent_chain_bus_listener()
    heartbeat_task = asyncio.create_task(heartbeat_loop(settings))

@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Shutting down Agent Chain...")
    if heartbeat_task:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


@app.get("/")
async def root() -> dict:
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "bus_enabled": settings.orion_bus_enabled,
        "request_channel": settings.agent_chain_request_channel,
        "result_prefix": settings.agent_chain_result_prefix,
    }


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
