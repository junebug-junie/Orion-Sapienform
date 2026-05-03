from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from orion.mind.v1 import MindRunRequestV1, MindRunResultV1

from .engine import run_mind_deterministic
from .settings import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("orion-mind")

app = FastAPI(title="Orion Mind", version=settings.SERVICE_VERSION)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "node": settings.NODE_NAME,
    }


@app.post("/v1/mind/run", response_model=MindRunResultV1)
async def mind_run(body: MindRunRequestV1) -> MindRunResultV1:
    router_dir: Path = settings.router_profiles_dir
    return run_mind_deterministic(
        body,
        router_profiles_dir=router_dir,
        snapshot_max_bytes=settings.MIND_SNAPSHOT_MAX_BYTES,
    )
