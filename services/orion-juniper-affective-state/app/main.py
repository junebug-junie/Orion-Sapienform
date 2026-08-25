"""Thin CPU orchestrator for the AffectGPT worker.

Manual/turn-scoped trigger only, still -- no ambient/background polling
loop. Two ways in as of 2026-08-22:

* ``POST /v1/juniper/affect/trigger`` -- point it at an already-written
  video+audio path pair (unchanged since this service's first version).
* ``POST /v1/juniper/affect/capture_and_assess`` -- the real cross-host
  bridge: bus RPC to orion-vision-retina (carbon) for a live clip, fetch the
  resulting percept-store blobs to a local temp dir here, then the same
  worker round-trip as above. This is what closes the loop from a live
  webcam/mic to a real assessment; Hub's toggle calls this endpoint.

Both paths converge on the same worker RPC and the same
``orion:affectgpt:assessment`` publish -- one real event stream regardless
of entry point.

Ambient/recurring capture DOES now exist (2026-08-22) -- Hub owns that loop
(services/orion-hub/scripts/vision_affect_ambient.py), not this service. It
just calls this same ``/capture_and_assess`` endpoint repeatedly with
``trigger="ambient"``; this service still has no scheduling logic of its
own and no notion of "on/off" -- every call here is still a single explicit
request from a caller, ambient or not. See ``JuniperMultimodalAffectV1``'s
``trigger``/``correlation_id`` fields (orion/schemas/affectgpt.py) for how a
consumer tells manual and ambient events apart and joins one attempt's
retina-RPC/worker-RPC/event legs together.
"""
import asyncio
import hashlib
import tempfile
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.schemas.affectgpt import (
    AffectGptAssessRequestPayload,
    AffectGptAssessResultPayload,
    JuniperMultimodalAffectV1,
)
from orion.schemas.vision import RetinaClipCaptureRequestPayload, RetinaClipCaptureResultPayload
from orion.situational.juniper_affect_state import write_latest_juniper_affect

from .settings import Settings

settings = Settings()

# Truncated excerpt only -- never the verbatim transcript. See
# orion/situational/juniper_affect_state.py's docstring and
# AffectContextV1 (orion/schemas/situation.py) for the privacy contract
# this feeds into a chat prompt.
_AFFECT_SUMMARY_MAX_CHARS = 300


class PerceptFetchError(Exception):
    """Raised by _fetch_percept -- unreachable store, HTTP error, or a hash
    mismatch on the fetched bytes (chain-of-custody check -- same discipline
    used to hand-verify carbon's first real capture live, 2026-08-22: never
    trust a reported sha256 without recomputing it from the bytes actually
    received)."""


def _normalize_trigger(value: object) -> Literal["manual", "ambient"]:
    """One place deciding what counts as a valid trigger label -- review
    finding, 2026-08-22: this clamp used to be duplicated verbatim in two
    call sites (capture_and_assess() and _wrap_event()), so a future third
    trigger value added to only one of them would silently miscategorize
    events with nothing to catch the mismatch."""
    return "ambient" if value == "ambient" else "manual"


class CaptureAndAssessRequest(BaseModel):
    """Real validation for POST /v1/juniper/affect/capture_and_assess's body
    -- review finding, 2026-08-22: this used to be an untyped ``dict``,
    silently coercing a malformed ``trigger`` (e.g. an int, a list) via
    _normalize_trigger deep inside application code instead of a 422 at the
    API boundary, unlike /trigger's own AffectGptAssessRequestPayload(**payload)
    validation."""

    model_config = ConfigDict(extra="ignore")

    subtitle: str = ""
    user_message: Optional[str] = None
    trigger: Literal["manual", "ambient"] = "manual"


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
        self,
        req: AffectGptAssessRequestPayload,
        *,
        trigger: str = "manual",
        corr_id: "uuid.UUID | None" = None,
    ) -> tuple[AffectGptAssessResultPayload, JuniperMultimodalAffectV1]:
        # Every path through this method falls through to the single
        # publish call at the bottom -- a real bug caught in review
        # (2026-08-22): the four early-return failure branches used to
        # `return` directly, so a bus-unavailable/timeout/empty-reply/decode
        # failure (the operationally most likely outcomes) never published
        # to orion:affectgpt:assessment at all, contradicting this class's
        # own README ("publishes ... success or failure").
        #
        # corr_id generated HERE if the caller didn't supply one (review
        # finding, 2026-08-22: /trigger's plain call used to leave this
        # None, so _call_worker generated its own internally for the real
        # RPC exchange but that id was never surfaced back to _wrap_event --
        # the published event's correlation_id was always None on this
        # path even though a real one existed and was used). Generating it
        # here, once, and threading it to BOTH _call_worker and _wrap_event
        # is what makes it real for every caller, not just capture_and_assess().
        corr_id = corr_id or uuid.uuid4()
        if not self.bus or not self.bus.enabled:
            result = AffectGptAssessResultPayload(
                ok=False, error="bus not connected", error_code="bus_unavailable"
            )
        else:
            result = await self._call_worker(req, corr_id=corr_id)

        event = self._wrap_event(result, req, trigger=trigger, corr_id=corr_id)
        await self._publish_event(event)
        return result, event

    async def _call_worker(
        self, req: AffectGptAssessRequestPayload, *, corr_id: "uuid.UUID | None" = None
    ) -> AffectGptAssessResultPayload:
        # corr_id is always supplied by trigger_assessment() above (it
        # generates one at the top if the caller didn't provide one) --
        # this fallback only matters for a hypothetical direct caller of
        # _call_worker() that skips trigger_assessment entirely.
        corr_id = corr_id or uuid.uuid4()
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

    async def capture_and_assess(
        self,
        subtitle: str = "",
        user_message: str | None = None,
        trigger: str = "manual",
    ) -> tuple[RetinaClipCaptureResultPayload, AffectGptAssessResultPayload, JuniperMultimodalAffectV1]:
        """Full loop: bus RPC to retina for a live clip -> fetch both blobs
        from percept-store into a local temp dir on circe -> the same
        worker round-trip trigger_assessment() already does -> the same
        published event. A capture failure is still a real, published
        assessment failure (error_code carries the actual stage) -- one
        observable failure surface for this whole capability, not a capture
        failure that silently never reaches orion:affectgpt:assessment.

        ``trigger`` distinguishes Hub's manual "Check now" button from its
        recurring ambient toggle (2026-08-22) -- both call this same method,
        only the label differs, so there's still one implementation, not two
        that can drift. Clamped to a known value rather than trusted
        verbatim: an unrecognized string here would otherwise reach
        JuniperMultimodalAffectV1's Literal field deep inside _wrap_event
        and raise a pydantic ValidationError with no handler around it.
        """
        trigger = _normalize_trigger(trigger)
        # ONE id for this whole attempt -- threaded through the retina RPC
        # below, the worker RPC inside trigger_assessment(), and the
        # published event, so all three legs of one tick are joinable via a
        # single id (Juniper's ask, 2026-08-22: "ensure the data model has
        # good ability to be correlative with other components in the mesh").
        corr_id = uuid.uuid4()
        capture = await self._capture_clip_via_retina(corr_id=corr_id)
        if not capture.ok:
            result = AffectGptAssessResultPayload(
                ok=False,
                error=f"capture failed: {capture.error}",
                error_code=capture.error_code or "capture_failed",
            )
            event = self._wrap_event(
                result,
                AffectGptAssessRequestPayload(video_path="", audio_path=""),
                trigger=trigger,
                corr_id=corr_id,
            )
            await self._publish_event(event)
            return capture, result, event

        # dir=settings.AFFECTGPT_SCRATCH_DIR, NOT the default /tmp -- that
        # default is private to THIS container. video_path/audio_path below
        # are resolved on the WORKER's filesystem (separate container); only
        # the shared scratch volume both containers mount is visible to it.
        # See AFFECTGPT_SCRATCH_DIR settings.py comment.
        Path(settings.AFFECTGPT_SCRATCH_DIR).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="orion-affect-capture-", dir=settings.AFFECTGPT_SCRATCH_DIR
        ) as tmpdir:
            video_path = str(Path(tmpdir) / "clip.mp4")
            audio_path = str(Path(tmpdir) / "clip.wav")
            # return_exceptions=True is load-bearing, not cosmetic: plain
            # gather() re-raises the moment ONE side fails while the other's
            # asyncio.to_thread call keeps running in a real OS thread (it
            # can't be cancelled). The bare `except` used to return from
            # inside this `with tempfile.TemporaryDirectory(...)` block right
            # then, so __exit__'s rmtree could race a sibling thread still
            # mid-write into the same directory -- review finding,
            # 2026-08-22. return_exceptions=True makes gather wait for BOTH
            # threads to actually finish before this line returns, so by the
            # time any error handling or cleanup runs, nothing is still
            # writing.
            fetch_results = await asyncio.gather(
                asyncio.to_thread(self._fetch_percept, capture.video_sha256, video_path),
                asyncio.to_thread(self._fetch_percept, capture.audio_sha256, audio_path),
                return_exceptions=True,
            )
            fetch_errors = [r for r in fetch_results if isinstance(r, BaseException)]
            if fetch_errors:
                detail = "; ".join(str(e) for e in fetch_errors)
                result = AffectGptAssessResultPayload(
                    ok=False, error=f"percept-store fetch failed: {detail}", error_code="fetch_failed"
                )
                event = self._wrap_event(
                    result,
                    AffectGptAssessRequestPayload(video_path="", audio_path=""),
                    trigger=trigger,
                    corr_id=corr_id,
                )
                await self._publish_event(event)
                return capture, result, event

            req = AffectGptAssessRequestPayload(
                video_path=video_path,
                audio_path=audio_path,
                subtitle=subtitle,
                user_message=user_message,
            )
            result, event = await self.trigger_assessment(req, trigger=trigger, corr_id=corr_id)
            # tmpdir (and the fetched clip bytes) is removed here on context
            # exit, always -- same "nothing survives past a temp dir"
            # discipline as retina's own clip_capture.py. The assess call
            # above has already read what it needs off disk by this point.
        return capture, result, event

    async def _capture_clip_via_retina(
        self, *, corr_id: "uuid.UUID | None" = None
    ) -> RetinaClipCaptureResultPayload:
        """Bus RPC twin of orion-vision-retina's POST /capture/clip. Mirrors
        _call_worker below almost exactly -- same RPC/decode/validate shape,
        different channel and payload types."""
        if not self.bus or not self.bus.enabled:
            return RetinaClipCaptureResultPayload(
                ok=False, error="bus not connected", error_code="bus_unavailable"
            )
        corr_id = corr_id or uuid.uuid4()
        reply_channel = f"{settings.CHANNEL_RETINA_CLIP_REPLY_PREFIX}:{corr_id}"
        request_envelope = BaseEnvelope(
            kind="retina.clip_capture.request",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            correlation_id=corr_id,
            reply_to=reply_channel,
            payload=RetinaClipCaptureRequestPayload(
                target_stream_id=settings.AFFECT_TARGET_STREAM_ID
            ).model_dump(),
        )
        try:
            msg = await self.bus.rpc_request(
                settings.CHANNEL_RETINA_CLIP_INTAKE,
                request_envelope,
                reply_channel=reply_channel,
                timeout_sec=settings.RETINA_CLIP_RPC_TIMEOUT_S,
            )
        except TimeoutError:
            return RetinaClipCaptureResultPayload(
                ok=False, error="retina did not reply in time", error_code="timeout"
            )

        data = msg.get("data") if isinstance(msg, dict) else None
        if not data:
            return RetinaClipCaptureResultPayload(
                ok=False, error="empty reply from retina", error_code="empty_reply"
            )

        decoded = self.bus.codec.decode(data)
        if not decoded.ok or not decoded.envelope:
            return RetinaClipCaptureResultPayload(
                ok=False, error=f"decode failed: {decoded.error}", error_code="decode_error"
            )

        try:
            payload = decoded.envelope.payload
            return (
                payload
                if isinstance(payload, RetinaClipCaptureResultPayload)
                else RetinaClipCaptureResultPayload(**payload)
            )
        except Exception as e:
            return RetinaClipCaptureResultPayload(
                ok=False, error=f"reply payload invalid: {e}", error_code="invalid_reply"
            )

    def _fetch_percept(self, sha256: str | None, dest_path: str) -> str:
        """GET {PERCEPT_STORE_BASE_URL}/{sha256}, verify the bytes actually
        hash to what was asked for, write to dest_path. Runs off the event
        loop via asyncio.to_thread (urllib is blocking) -- same convention
        as orion-vision-retina/app/frame_store.py's upload_bytes, and same
        "stdlib only, no new dependency for an HTTP GET" call (AGENTS.md
        section 10).
        """
        if not sha256:
            raise PerceptFetchError("no sha256 in capture result")
        base = str(settings.PERCEPT_STORE_BASE_URL or "").strip().rstrip("/")
        if not base:
            raise PerceptFetchError("PERCEPT_STORE_BASE_URL is unset")
        url = f"{base}/{sha256}"
        req = urllib.request.Request(url, method="GET")
        if settings.PERCEPT_STORE_TOKEN:
            req.add_header("X-Orion-Percept-Token", settings.PERCEPT_STORE_TOKEN)
        try:
            with urllib.request.urlopen(req, timeout=settings.PERCEPT_STORE_TIMEOUT_SEC) as resp:
                data = resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise PerceptFetchError(f"fetch {sha256[:12]} failed: {exc}") from exc
        fetched_sha = hashlib.sha256(data).hexdigest()
        if fetched_sha != sha256:
            raise PerceptFetchError(
                f"hash mismatch for {sha256[:12]}: store returned bytes hashing to {fetched_sha[:12]}"
            )
        with open(dest_path, "wb") as f:
            f.write(data)
        return dest_path

    def _wrap_event(
        self,
        result: AffectGptAssessResultPayload,
        req: AffectGptAssessRequestPayload,
        *,
        trigger: str = "manual",
        corr_id: "uuid.UUID | None" = None,
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
            subtitle_source=result.subtitle_source,
            transcript=result.transcript,
            input_ref={"video_path": req.video_path, "audio_path": req.audio_path},
            trigger=_normalize_trigger(trigger),
            correlation_id=str(corr_id) if corr_id else None,
        )

    async def _publish_event(self, event: JuniperMultimodalAffectV1):
        if not self.bus:
            return
        # BaseEnvelope's own correlation_id (the standard, documented
        # mesh-wide join key -- orion/core/bus/bus_schemas.py: "Stable id
        # for a request/flow") must carry the SAME id as the payload's own
        # correlation_id field, not an unrelated fresh uuid4() from
        # BaseEnvelope's default_factory -- review finding, 2026-08-22: a
        # consumer that already knows the standard envelope-level
        # convention (rather than digging into this payload's own nested
        # field) got a random id every time, defeating the whole point of
        # adding correlation_id here.
        envelope_kwargs = {}
        if event.correlation_id:
            envelope_kwargs["correlation_id"] = event.correlation_id
        envelope = BaseEnvelope(
            kind="affectgpt.juniper_multimodal_affect.v1",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=event.model_dump(mode="json"),
            **envelope_kwargs,
        )
        await self.bus.publish(settings.CHANNEL_AFFECTGPT_ASSESSMENT, envelope)

        # Mirror a successful read into the single-key store
        # orion/situational/context.py polls for chat-turn grounding
        # (2026-08-25 -- closes the loop this event stream previously
        # dead-ended at: nothing besides a manual debug CLI ever consumed
        # `orion:affectgpt:assessment` before this). Best-effort and
        # additive to the publish above, never a precondition for it --
        # this call sits after the real publish already succeeded, and
        # `write_latest_juniper_affect` is itself fail-open (never raises).
        # Skipped on a failed/empty capture: a failure should not overwrite
        # a real prior read, and the reader's own TTL/max-age gate already
        # ages that prior read out on its own schedule.
        if event.ok and event.raw_response:
            summary = event.raw_response.strip()
            if len(summary) > _AFFECT_SUMMARY_MAX_CHARS:
                summary = summary[: _AFFECT_SUMMARY_MAX_CHARS - 1] + "…"
            await write_latest_juniper_affect(
                self.bus,
                summary=summary,
                observed_at=event.observed_at,
                trigger=event.trigger,
                subtitle_source=event.subtitle_source,
            )


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


@app.post("/v1/juniper/affect/capture_and_assess")
async def capture_and_assess(payload: dict | None = None):
    """Live-capture entry point -- this is what Hub's "Check now" button AND
    its ambient toggle both call.

    No required body. Optional ``{"subtitle": "...", "user_message": "...",
    "trigger": "manual" | "ambient"}`` -- subtitle/user_message are the same
    fields /trigger's request has, minus the paths (those are filled in
    internally from the live capture, not caller-supplied). ``trigger``
    defaults to "manual"; Hub's ambient loop passes "ambient" explicitly so
    the published event can tell the two apart.

    Synchronous and can take up to ~195s worst case: retina's capture (~8s clip +
    RETINA_CLIP_TIMEOUT_SEC ceiling + RPC overhead) followed by the worker's
    own inference (README: ~20s warm, more cold). Callers need a generous
    timeout -- a slow reply here is normal, not a hang.
    """
    try:
        req = CaptureAndAssessRequest(**(payload or {}))
    except Exception as e:
        # Real validation at the API boundary (review finding, 2026-08-22)
        # -- matches /trigger's own AffectGptAssessRequestPayload(**payload)
        # pattern instead of silently coercing a malformed trigger deep
        # inside application code.
        return JSONResponse({"ok": False, "error": f"invalid request: {e}"}, status_code=422)

    capture, result, event = await service.capture_and_assess(
        subtitle=req.subtitle, user_message=req.user_message, trigger=req.trigger
    )
    return {
        "capture": capture.model_dump(exclude_none=True),
        "result": result.model_dump(),
        "event": event.model_dump(mode="json"),
    }
