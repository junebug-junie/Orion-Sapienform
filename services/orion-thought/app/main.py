from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .bus_listener import run_bus_worker
from .settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="[ORION-THOUGHT] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("orion-thought.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting orion-thought service=%s v=%s port=%s",
        settings.service_name,
        settings.service_version,
        settings.port,
    )
    app.state.bus_stop_event = asyncio.Event()
    app.state.bus_task = asyncio.create_task(run_bus_worker(app.state.bus_stop_event))
    yield
    app.state.bus_stop_event.set()
    task = app.state.bus_task
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="Orion Thought", lifespan=lifespan, version=settings.service_version)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": settings.service_name,
            "version": settings.service_version,
            "bus_enabled": settings.orion_bus_enabled,
            "channel_thought_request": settings.channel_thought_request,
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"service": settings.service_name, "status": "ok"})
