from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from . import lane_control
from .settings import settings

heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(f"[HOST] system_health_heartbeat_started service={settings.SERVICE_NAME}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[HOST] system_health_heartbeat_start_failed error={exc}")
        heartbeat_chassis = None
    yield
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HOST] system_health_heartbeat_stop_error error={exc}")


app = FastAPI(title="Orion GPU Lane Controller", version=settings.SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}


@app.get("/v1/gpu-lane/status")
async def get_status():
    # get_status() shells out to `docker compose ps` twice, synchronously
    # (subprocess.run with a 30s timeout each) -- uvicorn runs this service
    # single-process/single-event-loop (no --workers), so calling it inline
    # would freeze every other concurrent request, including /health, for
    # up to 60s on a slow or hung docker call. Review finding, fixed here.
    return await asyncio.to_thread(lane_control.get_status)


class GpuLaneFlipRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Literal["affect", "agent"]


def _authorized(authorization: Optional[str]) -> bool:
    # Fail closed: an unset GPU_LANE_CONTROLLER_TOKEN rejects every flip
    # request rather than disabling auth -- an operator who forgot to set
    # the token gets a clear 503, not a silently-open control-plane route.
    token = str(settings.GPU_LANE_CONTROLLER_TOKEN or "").strip()
    if not token:
        return False
    presented = str(authorization or "").strip()
    if presented.lower().startswith("bearer "):
        presented = presented[7:].strip()
    return secrets.compare_digest(presented, token)


@app.post("/v1/gpu-lane/flip")
async def flip(req: GpuLaneFlipRequest, authorization: Optional[str] = Header(default=None)):
    if not str(settings.GPU_LANE_CONTROLLER_TOKEN or "").strip():
        return JSONResponse(
            {"ok": False, "error": "GPU_LANE_CONTROLLER_TOKEN is not set on this service -- flip disabled"},
            status_code=503,
        )
    if not _authorized(authorization):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    result = await lane_control.flip(req.target)
    status = result.get("status")
    if status == "busy":
        return JSONResponse(result, status_code=409)
    ok = status in ("success", "noop")
    return JSONResponse(result, status_code=200 if ok else 502)
