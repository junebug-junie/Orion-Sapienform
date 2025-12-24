from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

# IMPORTANT: use the REAL Orion bus
from orion.core.bus.service import OrionBus

from .bus_worker import BusWorker
from .models import VisionResult, VisionTask
from .profiles import VisionProfiles
from .runner import VisionRunner
from .scheduler import VisionScheduler
from .settings import Settings

app = FastAPI(title="Orion Vision Host", version="0.1.0")

settings = Settings()

bus: Optional[OrionBus] = None
profiles: Optional[VisionProfiles] = None
runner: Optional[VisionRunner] = None
sched: Optional[VisionScheduler] = None

bus_worker: Optional[BusWorker] = None


def _apply_env_runtime() -> None:
    # Route HF caches to mounted dirs
    os.environ.setdefault("HF_HOME", settings.HF_HOME)
    os.environ.setdefault("TRANSFORMERS_CACHE", settings.TRANSFORMERS_CACHE)
    # Torch allocator config is injected by compose; don't stomp it here.


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "bus_enabled": bool(bus.enabled) if bus else False,
        "intake": settings.CHANNEL_VISIONHOST_INTAKE,
        "pub": settings.CHANNEL_VISIONHOST_PUB,
        "scheduler": {
            "max_inflight": settings.VISION_MAX_INFLIGHT,
            "max_inflight_per_gpu": settings.VISION_MAX_INFLIGHT_PER_GPU,
            "queue_when_busy": settings.VISION_QUEUE_WHEN_BUSY,
            "max_queue": settings.VISION_MAX_QUEUE,
        },
    }


@app.get("/profiles")
async def profiles_summary():
    if profiles is None:
        return JSONResponse({"ok": False, "error": "profiles not loaded"}, status_code=503)
    return {
        "ok": True,
        "version": profiles.version,
        "enabled": settings.enabled_profiles,
        "pipelines": list(profiles.pipelines.keys()),
        "profiles": list(profiles.profiles.keys()),
        "task_routing": profiles.task_routing,
    }


@app.post("/v1/vision/task")
async def http_task(payload: Dict[str, Any]):
    """
    Optional HTTP entrypoint.
    Minimal request:
      { "task_type": "...", "request": {...} }
    """
    if runner is None or sched is None:
        return JSONResponse({"ok": False, "error": "service not ready"}, status_code=503)

    corr_id = payload.get("corr_id") or str(uuid.uuid4())
    task_type = payload.get("task_type") or "retina_fast"
    request = payload.get("request") or {}
    reply_channel = payload.get("reply_channel") or f"{settings.CHANNEL_VISIONHOST_REPLY_PREFIX}:{corr_id}"

    task = VisionTask(
        corr_id=corr_id,
        reply_channel=reply_channel,
        task_type=task_type,
        request=request,
        meta=payload.get("meta") or {},
    )

    async def handler(pick):
        # If scheduler couldn't find a safe GPU, refuse (we're not pretending CPU can do retina).
        if pick.device == "cpu":
            return VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=None,
                error="No GPU available above hard floor (VRAM pressure).",
            )
        return await asyncio.to_thread(runner.execute, task, pick.device)

    try:
        res: VisionResult = await sched.submit(handler)
        return res.model_dump()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "corr_id": corr_id}, status_code=500)


async def _handle_bus_payload(payload: Dict[str, Any], source_channel: str) -> None:
    """
    Runs inside the asyncio loop (scheduled by BusWorker thread).
    """
    global bus, runner, sched
    if bus is None or runner is None or sched is None:
        return

    corr_id = payload.get("corr_id")
    task_type = payload.get("task_type")
    request = payload.get("request", {}) or {}
    reply_channel = payload.get("reply_channel")

    if not corr_id or not task_type:
        logger.warning(f"[INTAKE] missing corr_id/task_type source={source_channel} keys={list(payload.keys())}")
        return

    if not reply_channel:
        reply_channel = f"{settings.CHANNEL_VISIONHOST_REPLY_PREFIX}:{corr_id}"

    task = VisionTask(
        corr_id=str(corr_id),
        reply_channel=str(reply_channel),
        task_type=str(task_type),
        request=request,
        meta=payload.get("meta", {}) or {},
    )

    async def handler(pick):
        if pick.device == "cpu":
            res = VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=None,
                error="No GPU available above hard floor (VRAM pressure).",
            )
        else:
            res = await asyncio.to_thread(runner.execute, task, pick.device)

        # publish using your OrionBus (sync). Don't block the loop.
        await asyncio.to_thread(bus.publish, task.reply_channel, res.model_dump())
        await asyncio.to_thread(bus.publish, settings.CHANNEL_VISIONHOST_PUB, res.model_dump())
        return res

    try:
        await sched.submit(handler)
    except Exception as e:
        err = VisionResult(
            corr_id=task.corr_id,
            ok=False,
            task_type=task.task_type,
            device=None,
            error=str(e),
        )
        await asyncio.to_thread(bus.publish, task.reply_channel, err.model_dump())
        await asyncio.to_thread(bus.publish, settings.CHANNEL_VISIONHOST_PUB, err.model_dump())


@app.on_event("startup")
async def startup():
    global bus, profiles, runner, sched, bus_worker

    logger.remove()
    logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

    _apply_env_runtime()

    # Profiles + runner
    profiles = VisionProfiles(settings.VISION_PROFILES_PATH)
    profiles.load()

    runner = VisionRunner(
        profiles=profiles,
        enabled_names=settings.enabled_profiles,
        cache_dir=settings.MODEL_CACHE_DIR,
    )

    # Scheduler
    sched = VisionScheduler(
        devices=settings.devices,
        pick_metric=settings.VISION_PICK_GPU_METRIC,
        max_inflight=settings.VISION_MAX_INFLIGHT,
        max_inflight_per_gpu=settings.VISION_MAX_INFLIGHT_PER_GPU,
        queue_when_busy=settings.VISION_QUEUE_WHEN_BUSY,
        max_queue=settings.VISION_MAX_QUEUE,
        reserve_mb=settings.VISION_VRAM_RESERVE_MB,
        soft_floor_mb=settings.VISION_VRAM_SOFT_FLOOR_MB,
        hard_floor_mb=settings.VISION_VRAM_HARD_FLOOR_MB,
    )
    await sched.start()

    warmed = runner.warm_profiles()
    logger.info(f"[WARM] warmed={warmed}")

    # REAL OrionBus (sync redis) + worker thread
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    if bus.enabled and bus.client:
        loop = asyncio.get_running_loop()
        bus_worker = BusWorker(
            bus=bus,
            channel=settings.CHANNEL_VISIONHOST_INTAKE,
            loop=loop,
            on_payload=_handle_bus_payload,
        )
        bus_worker.start()
        logger.info(f"[READY] bus-first intake={settings.CHANNEL_VISIONHOST_INTAKE}")
    else:
        logger.warning("[READY] bus disabled or not connected (HTTP only)")


@app.on_event("shutdown")
async def shutdown():
    global bus_worker, sched, bus

    if bus_worker:
        try:
            bus_worker.stop()
        except Exception:
            pass
        bus_worker = None

    if sched:
        try:
            await sched.stop()
        except Exception:
            pass
        sched = None

    # OrionBus has no explicit close; redis client is managed internally
    bus = None
