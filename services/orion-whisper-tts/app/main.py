# services/orion-whisper-tts/app/main.py

import logging
import asyncio
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from orion.core.bus.async_service import OrionBusAsync
from .settings import settings
from .tts_worker import listener_worker

logging.basicConfig(
    level=logging.INFO,
    format="[WHISPER-TTS] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("orion-whisper-tts")

app = FastAPI(title="Orion Whisper/TTS Service")

# Global bus instance
bus: Optional[OrionBusAsync] = None
listener_task: Optional[asyncio.Task] = None


@app.on_event("startup")
async def startup() -> None:
    global bus, listener_task
    logger.info(
        "Starting Whisper/TTS service %s v%s",
        settings.service_name,
        settings.service_version,
    )

    bus = OrionBusAsync(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )
    await bus.connect()

    # Start the bus listener as an async task
    listener_task = asyncio.create_task(listener_worker(bus))


@app.on_event("shutdown")
async def shutdown() -> None:
    global bus, listener_task
    logger.info("Shutting down Whisper/TTS service...")

    if listener_task:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass

    if bus:
        await bus.close()


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.service_version,
            "bus": "connected" if (bus and bus.redis) else "disconnected",
        }
    )
