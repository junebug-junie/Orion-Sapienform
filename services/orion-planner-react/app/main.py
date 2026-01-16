# services/orion-planner-react/app/main.py

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1
from .settings import settings
from .api import router as planner_router

# Ensure bus_listener.py exists in the same folder!
from .bus_listener import start_planner_bus_listener_background

logger = logging.getLogger("planner-react.main")

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
                        node="planner-react-node",
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ─────────────────────────────────────────────────────────────
    # STARTUP: Launch the Bus Listener Thread
    # ─────────────────────────────────────────────────────────────
    logger.info("Starting Planner Bus Listener...")
    try:
        start_planner_bus_listener_background()
    except Exception as e:
        logger.error("Failed to start bus listener: %s", e)

    heartbeat_task = asyncio.create_task(heartbeat_loop(settings))

    yield

    # SHUTDOWN
    logger.info("Shutting down Planner React...")
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    lifespan=lifespan,  # <--- CRITICAL: This line starts the thread
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": settings.service_version,
    }


app.include_router(planner_router, prefix="")
