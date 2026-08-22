"""Thin CPU orchestrator for the AffectGPT worker.

Non-goal, deliberately: no live capture, no ambient/background polling loop.
There is currently no pipeline anywhere in this repo that captures Juniper's
webcam/mic -- building an "ambient mode" that polls for input that never
arrives would be exactly the empty-shell cognition CLAUDE.md bans. This
service exposes a manual/turn-scoped trigger: point it at an already-written
video+audio pair and it does the real bus round-trip to the worker, wraps
the result, and publishes a real domain event. Add ambient mode once a real
capture source exists to drive it.
"""
import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.schemas.affectgpt import (
    AffectGptAssessRequestPayload,
    AffectGptAssessResultPayload,
    JuniperMultimodalAffectV1,
)

from .settings import Settings

settings = Settings()


class JuniperAffectiveStateService:
    def __init__(self):
        self.bus: Optional[OrionBusAsync] = None

    async def start(self):
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)
        if settings.ORION_BUS_ENABLED:
            self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
            await self.bus.connect()
            logger.info("[READY] bus connected")
        else:
            logger.warning("[READY] bus disabled -- cannot reach the worker at all")

    async def stop(self):
        if self.bus:
            await self.bus.close()

    async def trigger_assessment(
        self, req: AffectGptAssessRequestPayload
    ) -> tuple[AffectGptAssessResultPayload, JuniperMultimodalAffectV1]:
        # Every path through this method falls through to the single
        # publish call at the bottom -- a real bug caught in review
        # (2026-08-22): the four early-return failure branches used to
        # `return` directly, so a bus-unavailable/timeout/empty-reply/decode
        # failure (the operationally most likely outcomes) never published
        # to orion:affectgpt:assessment at all, contradicting this class's
        # own README ("publishes ... success or failure").
        if not self.bus or not self.bus.enabled:
            result = AffectGptAssessResultPayload(
                ok=False, error="bus not connected", error_code="bus_unavailable"
            )
        else:
            result = await self._call_worker(req)

        event = self._wrap_event(result, req)
        await self._publish_event(event)
        return result, event

    async def _call_worker(
        self, req: AffectGptAssessRequestPayload
    ) -> AffectGptAssessResultPayload:
        corr_id = uuid.uuid4()
        reply_channel = f"{settings.CHANNEL_AFFECTGPT_REPLY_PREFIX}:{corr_id}"
        request_envelope = BaseEnvelope(
            kind="affectgpt.assess.request",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            correlation_id=corr_id,
            reply_to=reply_channel,
            payload=req.model_dump(),
        )

        try:
            msg = await self.bus.rpc_request(
                settings.CHANNEL_AFFECTGPT_INTAKE,
                request_envelope,
                reply_channel=reply_channel,
                timeout_sec=settings.AFFECTGPT_RPC_TIMEOUT_S,
            )
        except TimeoutError:
            return AffectGptAssessResultPayload(
                ok=False, error="worker did not reply in time", error_code="timeout"
            )

        data = msg.get("data") if isinstance(msg, dict) else None
        if not data:
            return AffectGptAssessResultPayload(
                ok=False, error="empty reply from worker", error_code="empty_reply"
            )

        decoded = self.bus.codec.decode(data)
        if not decoded.ok or not decoded.envelope:
            return AffectGptAssessResultPayload(
                ok=False, error=f"decode failed: {decoded.error}", error_code="decode_error"
            )

        try:
            payload = decoded.envelope.payload
            return (
                payload
                if isinstance(payload, AffectGptAssessResultPayload)
                else AffectGptAssessResultPayload(**payload)
            )
        except Exception as e:
            return AffectGptAssessResultPayload(
                ok=False, error=f"reply payload invalid: {e}", error_code="invalid_reply"
            )

    def _wrap_event(
        self, result: AffectGptAssessResultPayload, req: AffectGptAssessRequestPayload
    ) -> JuniperMultimodalAffectV1:
        return JuniperMultimodalAffectV1(
            observed_at=datetime.now(timezone.utc),
            ok=result.ok,
            raw_response=result.raw_response,
            error=result.error,
            error_code=result.error_code,
            model_ckpt=result.model_ckpt,
            face_detection=result.face_detection,
            timings=result.timings,
            input_ref={"video_path": req.video_path, "audio_path": req.audio_path},
        )

    async def _publish_event(self, event: JuniperMultimodalAffectV1):
        if not self.bus:
            return
        envelope = BaseEnvelope(
            kind="affectgpt.juniper_multimodal_affect.v1",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=event.model_dump(mode="json"),
        )
        await self.bus.publish(settings.CHANNEL_AFFECTGPT_ASSESSMENT, envelope)


service = JuniperAffectiveStateService()
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


app = FastAPI(
    title="Orion Juniper Affective State", version=settings.SERVICE_VERSION, lifespan=lifespan
)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "bus_enabled": bool(service.bus and service.bus.enabled),
    }


@app.post("/v1/juniper/affect/trigger")
async def trigger(payload: dict):
    """Manual/turn-scoped trigger -- no ambient mode, see module docstring."""
    try:
        req = AffectGptAssessRequestPayload(**payload)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"invalid request: {e}"}, status_code=422)

    result, event = await service.trigger_assessment(req)
    return {"result": result.model_dump(), "event": event.model_dump(mode="json")}
