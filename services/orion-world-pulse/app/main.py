from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from orion.autonomy.goal_state_listener import start_goal_state_listener, stop_goal_state_listener
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from app.routers.publish import router as publish_router
from app.routers.runs import router as runs_router
from app.settings import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion-world-pulse.main")

heartbeat_chassis: HeartbeatOnly | None = None
_goal_state_bus: OrionBusAsync | None = None
_goal_state_task: asyncio.Task[None] | None = None
_goal_state_stop: asyncio.Event | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from any bus usage inside the
    run/publish routers (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
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
    global heartbeat_chassis, _goal_state_bus, _goal_state_task, _goal_state_stop
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.service_name,
            settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None
    if settings.orion_bus_enabled:
        # Own, independent bus connection (mirrors build_heartbeat_chassis' rationale)
        # keeping this process's orion.autonomy.goal_state cache current -- SSP §6
        # Objective 6 (2026-07-30). Real caller: app/services/curiosity.py's
        # evaluate_capability() gate.
        try:
            _goal_state_bus = OrionBusAsync(url=settings.orion_bus_url)
            await _goal_state_bus.connect()
            _goal_state_stop = asyncio.Event()
            _goal_state_task = await start_goal_state_listener(
                _goal_state_bus, _goal_state_stop, channel=settings.channel_goal_proposal
            )
            logger.info("goal_state_listener_started channel=%s", settings.channel_goal_proposal)
        except Exception as exc:
            logger.warning("goal_state_listener_start_failed error=%s", exc)
            _goal_state_bus = None
            _goal_state_task = None
    yield
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error=%s", exc)
        heartbeat_chassis = None
    if _goal_state_stop is not None:
        _goal_state_stop.set()
    if _goal_state_task is not None:
        await stop_goal_state_listener(_goal_state_task)
        _goal_state_task = None
    if _goal_state_bus is not None:
        try:
            await _goal_state_bus.close()
        except Exception as exc:
            logger.warning("goal_state_listener_bus_close_error error=%s", exc)
        _goal_state_bus = None


app = FastAPI(title=settings.service_name, version=settings.service_version, lifespan=lifespan)
app.include_router(runs_router)
app.include_router(publish_router)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "node": settings.node_name,
        "enabled": settings.world_pulse_enabled,
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
