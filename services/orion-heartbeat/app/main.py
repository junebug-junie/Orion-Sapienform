from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .service import HeartbeatService
from .settings import settings

logging.basicConfig(level=logging.INFO, format="[orion-heartbeat] %(levelname)s - %(message)s")

svc = HeartbeatService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start_background()
    try:
        yield
    finally:
        await svc.stop()


app = FastAPI(title="orion-heartbeat", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.service_name,
        **svc.stats(),
    }


@app.get("/h1")
async def h1() -> dict:
    """Latest H1 (boundary/bulk entanglement) result -- None until the first
    h1_interval_sec tick has elapsed since service start. See
    app/substrate/reconstruction.py's module docstring for what this number
    means and why it differs from the 2026-05-01 charter's literal H1
    formula.
    """
    result = svc.latest_h1_dict()
    if result is None:
        return {"ok": False, "reason": "no_h1_computed_yet"}
    return {"ok": True, "h1": result}
