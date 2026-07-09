from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from datetime import datetime, timezone

from .bus_listener import run_bus_worker
from .chain import run_reverie_chain_worker
from .reasoning_activity import run_reasoning_worker
from .reasoning_activity import store as reasoning_store
from .reverie import run_reverie_worker
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
    # Spontaneous-thought mode — no-op unless ORION_REVERIE_ENABLED (default off).
    app.state.reverie_stop_event = asyncio.Event()
    app.state.reverie_task = asyncio.create_task(run_reverie_worker(app.state.reverie_stop_event))
    # Reverie chain mode — no-op unless ORION_REVERIE_CHAIN_ENABLED (default off).
    app.state.reverie_chain_stop_event = asyncio.Event()
    app.state.reverie_chain_task = asyncio.create_task(
        run_reverie_chain_worker(app.state.reverie_chain_stop_event)
    )
    # Reasoning-activity projection — always-on consumer. Harmless (empty
    # projection) when no producer is publishing reasoning_call events.
    app.state.reasoning_stop_event = asyncio.Event()
    app.state.reasoning_task = asyncio.create_task(
        run_reasoning_worker(app.state.reasoning_stop_event)
    )
    yield
    app.state.bus_stop_event.set()
    app.state.reverie_stop_event.set()
    app.state.reverie_chain_stop_event.set()
    app.state.reasoning_stop_event.set()
    for task in (
        app.state.bus_task,
        app.state.reverie_task,
        app.state.reverie_chain_task,
        app.state.reasoning_task,
    ):
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


@app.get("/projections/reasoning_activity")
async def reasoning_activity() -> JSONResponse:
    projection = reasoning_store.snapshot(datetime.now(timezone.utc))
    return JSONResponse({"ok": True, "projection": projection.model_dump(mode="json")})


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"service": settings.service_name, "status": "ok"})
