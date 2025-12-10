# FILE: services/orion-agent-chain/app/main.py
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .settings import settings
from .bus_listener import start_agent_chain_bus_listener

logging.basicConfig(
    level=logging.INFO,
    format="[AGENT-CHAIN] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("agent-chain.main")

app = FastAPI(
    title="Orion Agent Chain",
    version=settings.service_version,
)


@app.on_event("startup")
def on_startup() -> None:
    logger.info(
        "Starting Agent Chain (service=%s v=%s, port=%d)",
        settings.service_name,
        settings.service_version,
        settings.port,
    )
    start_agent_chain_bus_listener()


@app.get("/")
async def root() -> dict:
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "bus_enabled": settings.orion_bus_enabled,
        "request_channel": settings.agent_chain_request_channel,
        "result_prefix": settings.agent_chain_result_prefix,
    }


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
