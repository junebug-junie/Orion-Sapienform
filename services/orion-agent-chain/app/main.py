# services/orion-agent-chain/app/main.py

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import settings
from .api import router as api_router
from .bus_listener import start_agent_chain_bus_listener

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("agent-chain.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ─────────────────────────────────────────────────────────────
    # STARTUP: Launch the UI/Bus Listener
    # ─────────────────────────────────────────────────────────────
    logger.info("Starting Agent Chain Bus Listener...")
    try:
        start_agent_chain_bus_listener()
    except Exception as e:
        logger.error("Failed to start bus listener: %s", e)
    
    yield
    
    logger.info("Shutting down Agent Chain...")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        lifespan=lifespan,  # <--- Hook logic here
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/chain", tags=["agent-chain"])

    return app


app = create_app()
