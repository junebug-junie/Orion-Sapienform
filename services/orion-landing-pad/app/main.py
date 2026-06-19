from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from orion.bus.consumer_readiness import bus_consumer_readiness_v1, check_bus_consumer_readiness
from orion.schemas.telemetry.system_health import BusConsumerReadinessV1

from .service import LandingPadService
from .settings import settings
from .web.main import mount_web

service = LandingPadService(settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(title="orion-landing-pad", lifespan=lifespan)
mount_web(app, store=service.store, settings=settings)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "service": settings.app_name, "version": settings.service_version, "node": settings.node_name}


@app.get("/ready")
async def ready() -> JSONResponse:
    if not service.bus.enabled:
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=service.settings.pad_rpc_request_channel,
            subscriber_count=0,
            dependency_status="unavailable",
            error="bus not connected",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    redis = getattr(service.bus, "redis", None)
    if redis is None:
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=service.settings.pad_rpc_request_channel,
            subscriber_count=0,
            dependency_status="unavailable",
            error="redis unavailable",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    async def _rpc_smoke() -> bool:
        task = getattr(service.rpc, "_task", None)
        return task is not None and not task.done()

    result = await check_bus_consumer_readiness(
        redis,
        intake_channel=service.settings.pad_rpc_request_channel,
        service_name=service.settings.app_name,
        health_channel=service.settings.orion_health_channel,
        heartbeat_ttl_sec=float(service.settings.heartbeat_interval_sec) * 3.0,
        check_heartbeat=True,
        rpc_smoke_fn=_rpc_smoke,
    )
    body = bus_consumer_readiness_v1(result, http_alive=True)
    status_code = 200 if body.ok else 503
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)


@app.get("/frame/latest")
async def latest_frame() -> Dict[str, Any]:
    frame = await service.store.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="frame_not_found")
    return frame.model_dump(mode="json")


@app.get("/frames")
async def frames(limit: int = 10) -> Dict[str, Any]:
    frames = await service.store.get_frames(limit=limit)
    return {"count": len(frames), "frames": [f.model_dump(mode="json") for f in frames]}


@app.get("/events/salient")
async def salient_events(limit: int = 20) -> Dict[str, Any]:
    events = await service.store.get_salient_events(limit=limit)
    return {"count": len(events), "events": [e.model_dump(mode="json") for e in events]}


@app.get("/tensor/latest")
async def latest_tensor() -> Dict[str, Any]:
    tensor = await service.store.get_latest_tensor()
    if tensor is None:
        raise HTTPException(status_code=404, detail="tensor_not_found")
    return tensor.model_dump(mode="json")
