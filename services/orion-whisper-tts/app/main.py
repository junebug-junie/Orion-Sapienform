# services/orion-whisper-tts/app/main.py

import logging
import asyncio
import os
import uuid
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1

from .settings import settings
from .tts_worker import listener_worker
from .stt_worker import stt_listener_worker

logging.basicConfig(
    level=logging.INFO,
    format="[WHISPER-TTS] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("orion-whisper-tts")

app = FastAPI(title="Orion Whisper/TTS Service")

# Global bus instance
bus: Optional[OrionBusAsync] = None
listener_task: Optional[asyncio.Task] = None
stt_task: Optional[asyncio.Task] = None
heartbeat_task: Optional[asyncio.Task] = None

# Generate a unique Boot ID for this process instance
BOOT_ID = str(uuid.uuid4())



def _require_cuda_or_die() -> None:
    if torch.version.cuda is None or not torch.backends.cuda.is_built():
        raise RuntimeError(
            f"FATAL: torch is not a CUDA build. torch={torch.__version__} torch.version.cuda={torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FATAL: CUDA build detected but CUDA is not available at runtime. "
            f"torch.version.cuda={torch.version.cuda} "
            f"NVIDIA_VISIBLE_DEVICES={os.getenv('NVIDIA_VISIBLE_DEVICES')} "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')} "
            "Check container GPU passthrough (/dev/nvidia* should exist)."
        )

# call this during startup before spinning workers


async def heartbeat_loop(bus_instance: OrionBusAsync):
    """Publishes a heartbeat every 30 seconds."""
    logger.info(f"Heartbeat loop started. boot_id={BOOT_ID}")
    try:
        while True:
            try:
                # FIX: Added boot_id and last_seen_ts to satisfy SystemHealthV1 schema
                payload = SystemHealthV1(
                    service=settings.service_name,
                    version=settings.service_version,
                    node="whisper-node",
                    status="ok",
                    boot_id=BOOT_ID,
                    last_seen_ts=time.time()
                ).model_dump(mode="json")

                await bus_instance.publish("orion:system:health", BaseEnvelope(
                    kind="system.health.v1",
                    source=ServiceRef(name=settings.service_name, version=settings.service_version),
                    payload=payload
                ))
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

            await asyncio.sleep(30)
    except asyncio.CancelledError:
        logger.info("Heartbeat loop stopping...")


@app.on_event("startup")
async def startup() -> None:
    global bus, listener_task, stt_task, heartbeat_task
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
    stt_task = asyncio.create_task(stt_listener_worker(bus))
    heartbeat_task = asyncio.create_task(heartbeat_loop(bus))


@app.on_event("shutdown")
async def shutdown() -> None:
    global bus, listener_task, stt_task, heartbeat_task
    logger.info("Shutting down Whisper/TTS service...")

    if listener_task:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass

    if stt_task:
        stt_task.cancel()
        try:
            await stt_task
        except asyncio.CancelledError:
            pass

    if heartbeat_task:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
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
            "boot_id": BOOT_ID,
            "bus": "connected" if (bus and bus.redis) else "disconnected",
        }
    )
