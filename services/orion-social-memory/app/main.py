from __future__ import annotations

import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .service import SocialMemoryService
from .settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="[SOCIAL_MEMORY] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("orion-social-memory")

service = SocialMemoryService(settings=settings)
heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from `service`'s own bus
    connection (see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
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
    global heartbeat_chassis
    await service.start()
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
    try:
        yield
    finally:
        if heartbeat_chassis is not None:
            try:
                await heartbeat_chassis.stop()
            except Exception as exc:
                logger.warning("system_health_heartbeat_stop_error error=%s", exc)
            heartbeat_chassis = None
        with contextlib.suppress(Exception):
            await service.stop()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "service": settings.service_name, "version": settings.service_version}


@app.get("/summary")
async def summary(
    platform: str = Query(...),
    room_id: str = Query(...),
    participant_id: str | None = Query(None),
) -> Dict[str, Any]:
    if not platform.strip() or not room_id.strip():
        raise HTTPException(status_code=400, detail="platform_and_room_id_required")
    return await service.get_summary(platform=platform.strip(), room_id=room_id.strip(), participant_id=participant_id)


@app.get("/inspection")
async def inspection(
    platform: str = Query(...),
    room_id: str = Query(...),
    participant_id: str | None = Query(None),
) -> Dict[str, Any]:
    if not platform.strip() or not room_id.strip():
        raise HTTPException(status_code=400, detail="platform_and_room_id_required")
    logging.getLogger("orion-social-memory").info(
        "social_inspection_request_served room_id=%s participant_id=%s",
        room_id.strip(),
        participant_id or "room",
    )
    return await service.get_inspection(platform=platform.strip(), room_id=room_id.strip(), participant_id=participant_id)
