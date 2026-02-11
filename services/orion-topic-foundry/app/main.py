from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers.capabilities import router as capabilities_router
from app.routers.datasets import router as datasets_router
from app.routers.introspect import router as introspect_router
from app.routers.models import router as models_router
from app.routers.runs import router as runs_router
from app.settings import settings
from app.services.readiness import readiness_payload
from app.services.drift import drift_daemon_loop
from app.storage.repository import ensure_tables


logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion-topic-foundry")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_tables()
    logger.info("Topic Foundry service starting")
    drift_task = None
    if settings.topic_foundry_drift_daemon:
        logger.info("Starting drift daemon")
        drift_task = asyncio.create_task(drift_daemon_loop())
    try:
        yield
    finally:
        if drift_task:
            drift_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drift_task


app = FastAPI(title="Orion Topic Foundry", version=settings.service_version, lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "node": settings.node_name,
    }


@app.get("/ready")
def ready():
    return readiness_payload()


app.include_router(datasets_router)
app.include_router(capabilities_router)
app.include_router(models_router)
app.include_router(runs_router)
app.include_router(introspect_router)
