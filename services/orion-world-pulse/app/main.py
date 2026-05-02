from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from app.routers.publish import router as publish_router
from app.routers.runs import router as runs_router
from app.settings import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(title=settings.service_name, version=settings.service_version)
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
