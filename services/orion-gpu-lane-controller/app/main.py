from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly, Hunter
from orion.schemas.gpu_pool import GPU_POOL_ACTUATE_REQUEST_CHANNEL

from . import actuator_bus, lane_control, gpu2, pool_fence
from orion.schemas.gpu_slot import GpuSlotRequestV1
from .settings import settings

heartbeat_chassis: HeartbeatOnly | None = None
actuator_chassis: Hunter | None = None


def _chassis_config() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
    )


def build_heartbeat_chassis() -> HeartbeatOnly:
    return HeartbeatOnly(_chassis_config())


def build_actuator_chassis() -> Hunter:
    """Stage 4.2: pool actuation intake. Concurrent handlers so a `status` or a second request is
    answered (busy) while a 15-minute load runs; admission itself is serialized in actuator_bus."""
    holder: dict = {}

    async def on_request(env):
        await actuator_bus.handle(env.payload, holder["publish"], env.correlation_id)

    hunter = Hunter(_chassis_config(), handler=on_request, patterns=[GPU_POOL_ACTUATE_REQUEST_CHANNEL],
                    concurrent_handlers=True)
    holder["publish"] = actuator_bus.bus_publisher(hunter.bus)
    return hunter


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
    global actuator_chassis
    if gpu2.pool_authority():
        try:
            interrupted = await asyncio.to_thread(pool_fence.recover_interrupted)
            if interrupted:
                logger.warning(f"[HOST] gpu2_pool_action_interrupted action_id={interrupted['action_id']}")
        except Exception as exc:  # noqa: BLE001 -- the fence re-reads (and fails closed) per request
            logger.error(f"[HOST] gpu2_pool_fence_recover_failed error={exc}")
    if settings.ORION_BUS_ENABLED:
        try:
            actuator_chassis = build_actuator_chassis()
            await actuator_chassis.start_background()
            logger.info(f"[HOST] gpu_pool_actuator_started authority={settings.GPU2_AUTHORITY} "
                        f"actuator={settings.GPU_POOL_ACTUATOR_NAME}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HOST] gpu_pool_actuator_start_failed error={exc}")
            actuator_chassis = None
    yield
    if actuator_chassis is not None:
        try:
            await actuator_chassis.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HOST] gpu_pool_actuator_stop_error error={exc}")
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


@app.get("/v1/gpu-slots/{slot}/status")
async def slot_status(slot: str):
    if slot == "circe-gpu1":
        return await asyncio.to_thread(lane_control.get_status)
    if slot == "circe-gpu2":
        return await gpu2.status()
    return JSONResponse({"error": "unknown_slot"}, status_code=404)


@app.post("/v1/gpu-slots/activate")
async def activate_slot(req: GpuSlotRequestV1, authorization: Optional[str] = Header(default=None)):
    # GPU2 uses the deployment network boundary and durable ownership fencing.
    # Preserve the existing GPU1 authentication contract.
    if req.slot == "circe-gpu1":
        if not settings.GPU_LANE_CONTROLLER_TOKEN:
            return JSONResponse({"error": "mutation_disabled"}, status_code=503)
        if not str(authorization or "").lower().startswith("bearer ") or not _authorized(authorization):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    result = await lane_control.flip(req.target) if req.slot == "circe-gpu1" else await gpu2.flip(req)
    return JSONResponse(result, status_code=200 if result.get("status") in {"success", "noop"} else
                        409 if result.get("status") == "busy" else 503)
