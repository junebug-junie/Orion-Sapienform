import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.schemas.affectgpt import AffectGptAssessRequestPayload, AffectGptAssessResultPayload

from .face_extract import extract_face_crops
from .model_runtime import AffectGptRuntime
from .settings import Settings

settings = Settings()


class AffectGptWorkerService:
    def __init__(self):
        self.bus: Optional[OrionBusAsync] = None
        self.runtime = AffectGptRuntime(settings)
        self._consumer_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        # One 7B model on one GPU: real concurrent inference is not useful
        # and AffectGPT's Chat object is not documented thread-safe. A single
        # lock is the honest limit, not a placeholder for a future scheduler
        # -- see README non-goals.
        self._inflight_lock = asyncio.Lock()

    async def start(self):
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        logger.info("[WARM] loading AffectGPT model (this takes ~30-60s)...")
        await asyncio.to_thread(self.runtime.load)
        logger.info(f"[WARM] loaded ckpt={self.runtime.ckpt_path}")

        if settings.ORION_BUS_ENABLED:
            self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
            await self.bus.connect()
            self._consumer_task = asyncio.create_task(self._consume_loop())
            logger.info(f"[READY] bus-first intake={settings.CHANNEL_AFFECTGPT_INTAKE}")
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
        if self.bus:
            await self.bus.close()

    def readiness_payload(self) -> dict:
        bus_ok = True
        if settings.ORION_BUS_ENABLED:
            bus_ok = bool(self.bus and self.bus.enabled)
        ready = self.runtime.loaded and bus_ok
        return {
            "ready": ready,
            "model_loaded": self.runtime.loaded,
            "bus_required": bool(settings.ORION_BUS_ENABLED),
            "bus_connected": bus_ok,
        }

    async def run_assessment(
        self, payload: AffectGptAssessRequestPayload
    ) -> AffectGptAssessResultPayload:
        if not self.runtime.loaded:
            return AffectGptAssessResultPayload(
                ok=False, error="model not loaded", error_code="not_ready"
            )

        async with self._inflight_lock:
            try:
                face_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        extract_face_crops,
                        payload.video_path,
                        margin=settings.AFFECTGPT_FACE_MARGIN,
                        min_size_px=settings.AFFECTGPT_FACE_MIN_SIZE_PX,
                    ),
                    timeout=settings.AFFECTGPT_REQUEST_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                return AffectGptAssessResultPayload(
                    ok=False, error="face extraction timed out", error_code="timeout"
                )
            except Exception as exc:  # noqa: BLE001
                return AffectGptAssessResultPayload(
                    ok=False, error=str(exc), error_code="face_extraction_error"
                )

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.runtime.assess,
                        face_result=face_result,
                        audio_path=payload.audio_path,
                        subtitle=payload.subtitle,
                        user_message=payload.user_message,
                    ),
                    timeout=settings.AFFECTGPT_REQUEST_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                return AffectGptAssessResultPayload(
                    ok=False,
                    error="inference timed out",
                    error_code="timeout",
                    face_detection=face_result.as_meta(),
                )

        return AffectGptAssessResultPayload(
            ok=result.ok,
            raw_response=result.raw_response,
            error=result.error,
            error_code=result.error_code,
            model_ckpt=result.model_ckpt,
            face_or_frame_mode=result.face_or_frame_mode,
            face_detection=result.face_detection,
            timings=result.timings or None,
        )

    async def _consume_loop(self):
        if not self.bus:
            return
        async with self.bus.subscribe(settings.CHANNEL_AFFECTGPT_INTAKE) as pubsub:
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
                        asyncio.create_task(self._handle_envelope(decoded.envelope))
                except Exception as e:
                    logger.error(f"[BUS] consumer error: {e}")
                    await asyncio.sleep(1)

    async def _handle_envelope(self, envelope: BaseEnvelope):
        try:
            if isinstance(envelope.payload, dict):
                payload = AffectGptAssessRequestPayload(**envelope.payload)
            elif isinstance(envelope.payload, AffectGptAssessRequestPayload):
                payload = envelope.payload
            else:
                logger.error(f"[BUS] unexpected payload type: {type(envelope.payload)}")
                return
        except Exception as e:
            logger.error(f"[BUS] payload validation failed: {e}")
            return

        result = await self.run_assessment(payload)
        await self._publish_result(result, envelope)

    async def _publish_result(
        self, result: AffectGptAssessResultPayload, source_envelope: BaseEnvelope
    ):
        if not self.bus:
            return
        host_ref = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)
        reply_envelope = source_envelope.derive_child(
            kind="affectgpt.assess.result", source=host_ref, payload=result
        )
        corr_id = str(source_envelope.correlation_id)
        reply_channel = (
            source_envelope.reply_to or f"{settings.CHANNEL_AFFECTGPT_REPLY_PREFIX}:{corr_id}"
        )
        await self.bus.publish(reply_channel, reply_envelope)


service = AffectGptWorkerService()
heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    await service.start()
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(f"[HOST] system_health_heartbeat_started service={settings.SERVICE_NAME}")
    except Exception as exc:
        logger.warning(f"[HOST] system_health_heartbeat_start_failed error={exc}")
        heartbeat_chassis = None
    yield
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning(f"[HOST] system_health_heartbeat_stop_error error={exc}")
    await service.stop()


app = FastAPI(title="Orion AffectGPT Worker", version=settings.SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "model_loaded": service.runtime.loaded,
        "bus_enabled": bool(service.bus and service.bus.enabled),
        "intake": settings.CHANNEL_AFFECTGPT_INTAKE,
    }


@app.get("/ready")
async def ready():
    body = service.readiness_payload()
    code = 200 if body.get("ready") else 503
    return JSONResponse(body, status_code=code)


@app.post("/v1/affect/assess")
async def http_assess(payload: dict):
    """Direct HTTP entrypoint mirroring the bus handler -- see README."""
    try:
        req = AffectGptAssessRequestPayload(**payload)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"invalid request: {e}"}, status_code=422)

    result = await service.run_assessment(req)
    return result.model_dump()
