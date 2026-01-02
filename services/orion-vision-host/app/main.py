from __future__ import annotations

import asyncio
import os
import uuid
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import (
    VisionTaskRequestPayload,
    VisionTaskResultPayload,
    VisionArtifactPayload,
    VisionArtifactOutputs,
    VisionObject,
    VisionCaption,
    VisionEmbedding,
)

from .models import VisionTask
from .profiles import VisionProfiles
from .runner import VisionRunner
from .scheduler import VisionScheduler
from .settings import Settings

settings = Settings()

class VisionHostService:
    def __init__(self):
        self.bus: Optional[OrionBusAsync] = None
        self.profiles: Optional[VisionProfiles] = None
        self.runner: Optional[VisionRunner] = None
        self.sched: Optional[VisionScheduler] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        self._apply_env_runtime()

        # Profiles + runner
        self.profiles = VisionProfiles(settings.VISION_PROFILES_PATH)
        self.profiles.load()

        self.runner = VisionRunner(
            profiles=self.profiles,
            enabled_names=settings.enabled_profiles,
            cache_dir=settings.MODEL_CACHE_DIR,
        )

        # Scheduler
        self.sched = VisionScheduler(
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
        await self.sched.start()

        warmed = self.runner.warm_profiles()
        logger.info(f"[WARM] warmed={warmed}")

        # Bus
        if settings.ORION_BUS_ENABLED:
            self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
            await self.bus.connect()
            self._consumer_task = asyncio.create_task(self._consume_loop())
            logger.info(f"[READY] bus-first intake={settings.CHANNEL_VISIONHOST_INTAKE}")
        else:
            logger.warning("[READY] bus disabled (HTTP only)")

    async def stop(self):
        self._shutdown_event.set()
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        if self.sched:
            await self.sched.stop()

        if self.bus:
            await self.bus.close()

    def _apply_env_runtime(self) -> None:
        os.environ.setdefault("HF_HOME", settings.HF_HOME)
        os.environ.setdefault("TRANSFORMERS_CACHE", settings.TRANSFORMERS_CACHE)

    async def _consume_loop(self):
        if not self.bus:
            return

        async with self.bus.subscribe(settings.CHANNEL_VISIONHOST_INTAKE) as pubsub:
            while not self._shutdown_event.is_set():
                try:
                    async for msg in self.bus.iter_messages(pubsub):
                        if self._shutdown_event.is_set():
                            break

                        data = msg.get("data")
                        if not data:
                            continue

                        decoded = self.bus.codec.decode(data)
                        if not decoded.ok or not decoded.envelope:
                            logger.error(f"[BUS] decode failed: {decoded.error}")
                            continue

                        # Process in background to not block consumer
                        asyncio.create_task(self._handle_envelope(decoded.envelope))

                except Exception as e:
                    logger.error(f"[BUS] consumer error: {e}")
                    await asyncio.sleep(1)

    async def _handle_envelope(self, envelope: BaseEnvelope):
        # Validate payload
        try:
            if isinstance(envelope.payload, dict):
                 # Try to convert dict to model if codec returned dict
                payload = VisionTaskRequestPayload(**envelope.payload)
            elif isinstance(envelope.payload, VisionTaskRequestPayload):
                payload = envelope.payload
            else:
                logger.error(f"[BUS] unexpected payload type: {type(envelope.payload)}")
                return
        except Exception as e:
            logger.error(f"[BUS] payload validation failed: {e}")
            return

        corr_id = envelope.correlation_id or str(uuid.uuid4())
        reply_to = envelope.reply_to or f"{settings.CHANNEL_VISIONHOST_REPLY_PREFIX}:{corr_id}"

        task = VisionTask(
            corr_id=corr_id,
            reply_channel=reply_to,
            task_type=payload.task_type,
            request=payload.request,
            meta=payload.meta or {},
        )

        async def handler(pick):
            if pick.device == "cpu":
                 return VisionResult(
                    corr_id=task.corr_id,
                    ok=False,
                    task_type=task.task_type,
                    device=None,
                    error="No GPU available above hard floor (VRAM pressure).",
                )

            # Execute on thread
            return await asyncio.to_thread(self.runner.execute, task, pick.device)

        try:
            res: VisionResult = await self.sched.submit(handler)
            await self._publish_result(res, envelope)
        except Exception as e:
            err = VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=None,
                error=str(e),
            )
            await self._publish_result(err, envelope)

    async def _publish_result(self, res: VisionResult, source_envelope: BaseEnvelope):
        if not self.bus:
            return

        # Prepare result payload
        result_payload = VisionTaskResultPayload(
            ok=res.ok,
            task_type=res.task_type,
            device=res.device,
            error=res.error,
            result=res.artifacts, # 'artifacts' in VisionResult holds the output dict
            artifact_id=None, # TODO: generate/extract artifact ID if consolidated
            timings=res.meta.get("timings") # or extract from meta
        )

        # Publish reply
        reply_envelope = BaseEnvelope(
            schema_id="vision.task.result",
            schema_version="1.0.0",
            kind="vision.task.result",
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=res.corr_id,
            causality_chain=source_envelope.causality_chain + [source_envelope.correlation_id] if source_envelope.correlation_id else [],
            payload=result_payload
        )
        await self.bus.publish(res.reply_channel, reply_envelope)

        if res.ok and res.artifacts:
             await self._publish_artifact_broadcast(res, source_envelope)

    async def _publish_artifact_broadcast(self, res: VisionResult, source_envelope: BaseEnvelope):
        # We need to construct a canonical VisionArtifactPayload
        # The runner returns a dict 'artifacts'. We need to parse/normalize it.

        # This is best effort mapping from the free-form runner output to the strict schema.
        artifacts = res.artifacts or {}

        # Generate a deterministic artifact ID if not present
        artifact_id = str(uuid.uuid4()) # In real world, hash inputs + model

        # Map outputs
        objects = None
        caption = None
        embedding = None

        if "objects" in artifacts and isinstance(artifacts["objects"], list):
            objects = []
            for obj in artifacts["objects"]:
                objects.append(VisionObject(
                    label=str(obj.get("label", "unknown")),
                    score=float(obj.get("score", 0.0)),
                    box_xyxy=obj.get("box_xyxy", [0,0,0,0])
                ))

        if "caption" in artifacts and isinstance(artifacts["caption"], dict):
            caption = VisionCaption(
                text=artifacts["caption"].get("text", ""),
                confidence=artifacts["caption"].get("confidence")
            )

        if "embedding" in artifacts and isinstance(artifacts["embedding"], dict):
            embedding = VisionEmbedding(
                ref=artifacts["embedding"].get("ref", ""),
                path=artifacts["embedding"].get("path", ""),
                dim=artifacts["embedding"].get("dim", 0)
            )

        outputs = VisionArtifactOutputs(
            objects=objects,
            caption=caption,
            embedding=embedding
        )
        # Add any extra fields from artifacts to outputs if needed, but schema says extra="allow"
        for k, v in artifacts.items():
            if k not in ("objects", "caption", "embedding", "configured", "implemented", "kind", "device", "model_id"):
                setattr(outputs, k, v)

        payload = VisionArtifactPayload(
            artifact_id=artifact_id,
            correlation_id=res.corr_id,
            task_type=res.task_type,
            device=res.device or "unknown",
            inputs=res.meta.get("request", {}) or {}, # We might need to pass request better
            outputs=outputs,
            timing={"latency_s": res.meta.get("latency_s", 0.0)},
            model_fingerprints={res.task_type: artifacts.get("model_id", "unknown")}
        )

        envelope = BaseEnvelope(
            schema_id="vision.artifact",
            schema_version="1.0.0",
            kind="vision.artifact",
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=res.corr_id,
            causality_chain=source_envelope.causality_chain + [source_envelope.correlation_id] if source_envelope.correlation_id else [],
            payload=payload
        )

        await self.bus.publish(settings.CHANNEL_VISIONHOST_PUB, envelope)

service = VisionHostService()
app = FastAPI(title="Orion Vision Host", version="0.1.0", lifespan=None) # We use on_event for now or convert to lifespan

@app.on_event("startup")
async def startup():
    await service.start()

@app.on_event("shutdown")
async def shutdown():
    await service.stop()

@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "bus_enabled": bool(service.bus and service.bus.enabled),
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
    if not service.profiles:
        return JSONResponse({"ok": False, "error": "profiles not loaded"}, status_code=503)
    return {
        "ok": True,
        "version": service.profiles.version,
        "enabled": settings.enabled_profiles,
        "pipelines": list(service.profiles.pipelines.keys()),
        "profiles": list(service.profiles.profiles.keys()),
        "task_routing": service.profiles.task_routing,
    }

@app.post("/v1/vision/task")
async def http_task(payload: Dict[str, Any]):
    """
    Optional HTTP entrypoint.
    Minimal request:
      { "task_type": "...", "request": {...} }
    """
    if not service.runner or not service.sched:
        return JSONResponse({"ok": False, "error": "service not ready"}, status_code=503)

    corr_id = payload.get("corr_id") or str(uuid.uuid4())
    task_type = payload.get("task_type") or "retina_fast"
    request = payload.get("request") or {}

    # We fake a VisionTaskRequestPayload
    task_payload = VisionTaskRequestPayload(
        task_type=task_type,
        request=request,
        meta=payload.get("meta")
    )

    # We mimic the bus handling logic but return result directly
    task = VisionTask(
        corr_id=corr_id,
        reply_channel="http-direct",
        task_type=task_payload.task_type,
        request=task_payload.request,
        meta=task_payload.meta or {},
    )

    async def handler(pick):
        if pick.device == "cpu":
            return VisionResult(
                corr_id=task.corr_id,
                ok=False,
                task_type=task.task_type,
                device=None,
                error="No GPU available above hard floor (VRAM pressure).",
            )
        return await asyncio.to_thread(service.runner.execute, task, pick.device)

    try:
        res: VisionResult = await service.sched.submit(handler)

        # Also broadcast artifact if success
        if res.ok and res.artifacts and service.bus:
             # We create a dummy source envelope
             dummy_env = BaseEnvelope(
                 schema_id="http", schema_version="1", kind="http", source="http", correlation_id=corr_id, payload={}
             )
             await service._publish_artifact_broadcast(res, dummy_env)

        return res.model_dump()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "corr_id": corr_id}, status_code=500)
