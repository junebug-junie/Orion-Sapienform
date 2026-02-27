# services/orion-agent-chain/app/main.py
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1

from .settings import settings
from .bus_listener import run_bus_worker

logging.basicConfig(
    level=logging.INFO,
    format="[AGENT-CHAIN] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("agent-chain.main")

BOOT_ID = str(uuid.uuid4())


async def heartbeat_loop(settings):
    """Publishes a heartbeat every 30 seconds."""
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
                        boot_id=BOOT_ID,
                        last_seen_ts=datetime.now(timezone.utc),
                        node="agent-chain-node",
                        status="ok",
                    ).model_dump(mode="json")

                    await bus.publish(
                        "orion:system:health",
                        BaseEnvelope(
                            kind="system.health.v1",
                            source=ServiceRef(name=settings.service_name, version=settings.service_version),
                            payload=payload,
                        ),
                    )
                except Exception as e:
                    logger.warning("Heartbeat failed: %s", e)

            await asyncio.sleep(30)
    except asyncio.CancelledError:
        logger.info("Heartbeat loop stopping...")
        raise
    finally:
        if bus.enabled:
            await bus.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting Agent Chain (service=%s v=%s, port=%s)",
        settings.service_name,
        settings.service_version,
        settings.port,
    )
    app.state.bus_stop_event = asyncio.Event()
    app.state.bus_task = asyncio.create_task(run_bus_worker(app.state.bus_stop_event), name="agent_chain_bus_worker")
    app.state.heartbeat_task = asyncio.create_task(heartbeat_loop(settings), name="agent_chain_heartbeat")
    logger.info("Agent-chain background tasks started (bus_worker + heartbeat)")
    try:
        yield
    finally:
        logger.info("Shutting down Agent Chain...")
        app.state.bus_stop_event.set()
        for task_name in ("bus_task", "heartbeat_task"):
            task = getattr(app.state, task_name, None)
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        logger.info("Agent-chain background tasks stopped")


app = FastAPI(
    title="Orion Agent Chain",
    version=settings.service_version,
    lifespan=lifespan,
)


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
