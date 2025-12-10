# services/orion-planner-react/app/main.py

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .settings import settings
from .api import router as planner_router

# Ensure bus_listener.py exists in the same folder!
from .bus_listener import start_planner_bus_listener_background

logger = logging.getLogger("planner-react.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ─────────────────────────────────────────────────────────────
    # STARTUP: Launch the Bus Listener Thread
    # ─────────────────────────────────────────────────────────────
    logger.info("Starting Planner Bus Listener...")
    try:
        start_planner_bus_listener_background()
    except Exception as e:
        logger.error("Failed to start bus listener: %s", e)

    yield

    # SHUTDOWN
    logger.info("Shutting down Planner React...")


app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    lifespan=lifespan,  # <--- CRITICAL: This line starts the thread
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": settings.service_version,
    }


app.include_router(planner_router, prefix="")
