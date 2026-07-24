from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .bus_listener import run_bus_worker
from .cancel_listener import run_cancel_worker
from .settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="[ORION-HARNESS-GOV] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("orion-harness-governor.main")


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from run_bus_worker/run_cancel_worker's
    own bus connections (see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=settings.orion_bus_enabled,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting orion-harness-governor service=%s v=%s port=%s",
        settings.service_name,
        settings.service_version,
        settings.port,
    )
    app.state.bus_stop_event = asyncio.Event()
    app.state.bus_task = asyncio.create_task(run_bus_worker(app.state.bus_stop_event))
    app.state.cancel_task = asyncio.create_task(run_cancel_worker(app.state.bus_stop_event))
    app.state.heartbeat_chassis = build_heartbeat_chassis()
    try:
        await app.state.heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.service_name,
            settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        app.state.heartbeat_chassis = None
    yield
    app.state.bus_stop_event.set()
    for task in (app.state.bus_task, getattr(app.state, "cancel_task", None)):
        if task is None:
            continue
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    heartbeat_chassis = getattr(app.state, "heartbeat_chassis", None)
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error=%s", exc)


app = FastAPI(title="Orion Harness Governor", lifespan=lifespan, version=settings.service_version)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": settings.service_name,
            "version": settings.service_version,
            "bus_enabled": settings.orion_bus_enabled,
            "governor_enabled": settings.orion_harness_governor_enabled,
            "channel_harness_run_request": settings.channel_harness_run_request,
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"service": settings.service_name, "status": "ok"})
