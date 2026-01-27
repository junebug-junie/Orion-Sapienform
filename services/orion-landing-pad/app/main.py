from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from loguru import logger

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
