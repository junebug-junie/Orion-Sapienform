from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from app.routers.ideation import router as ideation_router
from app.routers.v1 import router as v1_router
from app.settings import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(title=settings.service_name, version=settings.service_version)
app.include_router(v1_router)
app.include_router(ideation_router)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "enabled": settings.knowledge_forge_enabled,
        "write_enabled": settings.knowledge_forge_write_enabled,
        "ideation_enabled": settings.knowledge_forge_ideation_enabled,
        "ideation_write_enabled": settings.knowledge_forge_ideation_write_enabled,
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
