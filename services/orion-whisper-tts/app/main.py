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
from .cuda_watchdog import cuda_watchdog_loop, restart_process

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
cuda_watchdog_task: Optional[asyncio.Task] = None

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
    global bus, listener_task, stt_task, heartbeat_task, cuda_watchdog_task
    logger.info(
        "Starting Whisper/TTS service %s v%s",
        settings.service_name,
        settings.service_version,
    )
    logger.info(
        "TTS configured backend=%s model=%s gpu=%s language=%s default_speaker=%s default_speaker_wav=%s",
        settings.tts_backend,
        settings.tts_model_name,
        settings.tts_use_gpu,
        settings.tts_default_language,
        settings.tts_default_speaker or "(none)",
        settings.tts_default_speaker_wav or "(none)",
    )

    gpu_expected = settings.tts_use_gpu

    # Advisory ONLY -- logs loud, never raises. Review finding, 2026-08-26:
    # an earlier version of this patch let _require_cuda_or_die() raise here,
    # which crashed the ENTIRE process -- bus, listener_task, stt_task, all
    # of it -- before any of them started. STT (openai-whisper) does not
    # need CUDA at all (stt.py falls back to CPU); the whole point of the
    # incident this patch closes is that STT survived a CUDA staleness event
    # that killed TTS. A hard crash-at-boot took away exactly the resilience
    # this closes on. Boot with CUDA already broken must degrade the same
    # way a break mid-uptime does -- loud logs, bus/STT keep running -- not
    # worse.
    #
    # Enforcement (actually acting on a real GPU failure) is the
    # cuda_watchdog's job alone, below -- ONE mechanism for both "broken at
    # boot" and "broken mid-uptime" rather than two that can drift out of
    # sync. A boot-broken GPU fails its first watchdog check almost
    # immediately (poll_sec after boot) and restarts on the configured
    # threshold, same as any other detected staleness.
    if gpu_expected:
        try:
            _require_cuda_or_die()
        except RuntimeError as exc:
            logger.critical("[WHISPER-TTS] cuda_unavailable_at_boot %s", exc)

    bus = OrionBusAsync(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )
    await bus.connect()

    # Start the bus listener as an async task
    listener_task = asyncio.create_task(listener_worker(bus))
    stt_task = asyncio.create_task(stt_listener_worker(bus))
    heartbeat_task = asyncio.create_task(heartbeat_loop(bus))
    if gpu_expected and settings.cuda_watchdog_enabled:
        cuda_watchdog_task = asyncio.create_task(
            cuda_watchdog_loop(
                poll_sec=settings.cuda_watchdog_poll_sec,
                failure_threshold=settings.cuda_watchdog_failure_threshold,
                is_cuda_available=torch.cuda.is_available,
                on_trigger=restart_process,
            )
        )


@app.on_event("shutdown")
async def shutdown() -> None:
    global bus, listener_task, stt_task, heartbeat_task, cuda_watchdog_task
    logger.info("Shutting down Whisper/TTS service...")

    # Review finding, 2026-08-26: these were four hand-copied cancel/await/
    # except blocks (three pre-dating this patch, one added by it) that
    # could drift independently -- a 5th background task added later means
    # another hand-applied copy, easy to apply to some sites and silently
    # miss one. One helper, applied uniformly to all four.
    for task in (listener_task, stt_task, cuda_watchdog_task, heartbeat_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    if bus:
        await bus.close()


@app.get("/health")
async def health():
    # Review finding, 2026-08-26: neither /health nor the heartbeat's own
    # bus publish reflected CUDA state at all -- for the full poll_sec *
    # failure_threshold window before a restart, both kept reporting
    # "status: ok" while the watchdog was silently counting toward one.
    # A fresh, direct read here (not the watchdog's own internal counter,
    # which stays private to that loop) gives an operator or monitor
    # polling /health real-time visibility without waiting on a log line.
    cuda_status = None
    if settings.tts_use_gpu:
        try:
            cuda_status = bool(torch.cuda.is_available())
        except Exception:
            cuda_status = False
    return JSONResponse(
        {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.service_version,
            "boot_id": BOOT_ID,
            "bus": "connected" if (bus and bus.redis) else "disconnected",
            "cuda_available": cuda_status,
            "cuda_watchdog_enabled": bool(
                settings.tts_use_gpu and settings.cuda_watchdog_enabled
            ),
        }
    )
