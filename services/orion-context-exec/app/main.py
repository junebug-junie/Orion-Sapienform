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

from .api import router, set_runner
from .proposal_review_api import router as proposal_review_router
from .bus_listener import run_bus_worker
from .runner import ContextExecRunner
from .settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="[CONTEXT-EXEC] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("context-exec.main")

BOOT_ID = str(uuid.uuid4())


async def heartbeat_loop() -> None:
    bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
    if bus.enabled:
        await bus.connect()
    try:
        while True:
            if bus.enabled:
                try:
                    payload = SystemHealthV1(
                        service=settings.service_name,
                        version=settings.service_version,
                        boot_id=BOOT_ID,
                        last_seen_ts=datetime.now(timezone.utc),
                        node=settings.node_name,
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
                except Exception as exc:
                    logger.warning("heartbeat failed: %s", exc)
            await asyncio.sleep(30)
    except asyncio.CancelledError:
        raise
    finally:
        if bus.enabled:
            await bus.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting context-exec service=%s v=%s port=%s",
        settings.service_name,
        settings.service_version,
        settings.port,
    )
    app.state.bus_stop_event = asyncio.Event()
    bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
    if bus.enabled:
        await bus.connect()
    app.state.bus = bus
    runner = ContextExecRunner(bus=bus if bus.enabled else None)
    set_runner(runner)
    app.state.bus_task = asyncio.create_task(run_bus_worker(app.state.bus_stop_event))
    app.state.heartbeat_task = asyncio.create_task(heartbeat_loop())
    yield
    app.state.bus_stop_event.set()
    for task in (app.state.bus_task, app.state.heartbeat_task):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    bus = getattr(app.state, "bus", None)
    if bus is not None and bus.enabled:
        with suppress(Exception):
            await bus.close()


app = FastAPI(title="Orion Context Exec", lifespan=lifespan)
app.include_router(router)
app.include_router(proposal_review_router)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"service": settings.service_name, "status": "ok"})
