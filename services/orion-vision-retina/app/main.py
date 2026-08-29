from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import (
    RetinaClipCaptureRequestPayload,
    RetinaClipCaptureResultPayload,
    VisionFramePointerPayload,
)

from .clip_capture import ClipCaptureCooldownError, ClipCaptureError, capture_clip
from .envelopes import make_frame_pointer_envelope
from .frame_store import (
    PerceptUploadError,
    cleanup_old_frames,
    save_frame,
    upload_bytes,
    upload_frame,
)
from .health import RetinaMetrics, make_system_health_envelope
from .settings import Settings, get_settings
from .sources import create_frame_source


class RetinaService:
    def __init__(
        self,
        settings: Settings | None = None,
        bus: OrionBusAsync | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bus = bus or OrionBusAsync(
            url=self.settings.ORION_BUS_URL,
            enforce_catalog=self.settings.ORION_BUS_ENFORCE_CATALOG,
        )
        self.source = create_frame_source(
            self.settings.RETINA_SOURCE_TYPE,
            self.settings.RETINA_SOURCE,
            width=self.settings.RETINA_WIDTH,
            height=self.settings.RETINA_HEIGHT,
            reconnect_seconds=self.settings.SOURCE_RECONNECT_SECONDS,
        )
        self.metrics = RetinaMetrics()
        self._capture_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        self._last_cleanup = 0.0
        self._started_at = time.time()
        self._last_publish_ts: float | None = None
        self._sample_attempted = False
        # Guards physical device access between the continuous capture_loop
        # and on-demand clip capture (app/clip_capture.py). Real bug found
        # live on carbon (2026-08-22): both open /dev/video0 -- the
        # continuous loop via cv2.VideoCapture (self.source), clip capture
        # via a separate ffmpeg process -- and a webcam only accepts one
        # exclusive handle at a time, so ffmpeg failed with "Device or
        # resource busy" every time. Nothing in review or this module's own
        # tests could have caught this: it only manifests with a real
        # camera device, which this session never had access to.
        self._device_lock = asyncio.Lock()
        # One capture at a time: /dev/video0 is exclusive-access, and a
        # second ffmpeg trying to open it mid-capture fails with an opaque
        # device-busy error instead of a clear "already in progress"
        # response. Shared by the HTTP route and the bus RPC consumer below
        # -- both must serialize through the same lock, not one each.
        self._clip_capture_lock = asyncio.Lock()
        self._clip_consumer_task: Optional[asyncio.Task] = None
        # 0.0 means "no prior capture" -- the very first request is never
        # subject to the cooldown. See RETINA_CLIP_MIN_INTERVAL_SEC.
        self._last_clip_capture_ts: float = 0.0

    async def start(self) -> None:
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=self.settings.LOG_LEVEL)

        await self.bus.connect()
        await self.source.start()
        self._shutdown.clear()
        self._capture_task = asyncio.create_task(self.capture_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        if self.settings.RETINA_CLIP_ENABLED:
            self._clip_consumer_task = asyncio.create_task(self._clip_consume_loop())
            logger.info(
                f"[RETINA] clip RPC intake → {self.settings.CHANNEL_RETINA_CLIP_INTAKE}"
            )
        logger.info(f"[RETINA] Started → {self.settings.CHANNEL_RETINA_PUB}")

    async def stop(self) -> None:
        self._shutdown.set()
        for task in (self._capture_task, self._health_task, self._clip_consumer_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.source.stop()
        await self.bus.close()

    async def capture_and_upload_clip(
        self, *, want_audio: bool = True
    ) -> RetinaClipCaptureResultPayload:
        """Shared body for both POST /capture/clip and the bus RPC consumer
        -- record a clip, upload video+audio to percept-store, return refs.
        Raises ClipCaptureError / PerceptUploadError / OSError; callers map
        those to their own transport's error shape (HTTP status vs a
        RetinaClipCaptureResultPayload with ok=False).
        """
        s = self.settings
        async with self._clip_capture_lock:
            # Re-checked here (inside the lock, not before acquiring it) so
            # a request that queued up behind an in-flight capture is still
            # subject to the cooldown measured from the PRIOR capture's
            # actual completion -- checking before the lock would let a
            # queued request fire the instant the lock frees, defeating the
            # whole point. See RETINA_CLIP_MIN_INTERVAL_SEC / review finding
            # 2026-08-22.
            elapsed = time.time() - self._last_clip_capture_ts
            if elapsed < s.RETINA_CLIP_MIN_INTERVAL_SEC:
                wait = s.RETINA_CLIP_MIN_INTERVAL_SEC - elapsed
                raise ClipCaptureCooldownError(
                    f"capture requested {elapsed:.1f}s after the last one finished; "
                    f"minimum interval is {s.RETINA_CLIP_MIN_INTERVAL_SEC:.0f}s "
                    f"({wait:.1f}s remaining)"
                )
            # pause_device(): release /dev/video0 for the continuous
            # capture_loop first -- see _device_lock comment above. Real bug
            # found live on carbon (2026-08-22): without this, ffmpeg failed
            # every time with "Device or resource busy" because capture_loop
            # already held the webcam open.
            async with self.pause_device():
                result = await capture_clip(
                    ffmpeg_bin=s.RETINA_CLIP_FFMPEG_BIN,
                    video_device=s.RETINA_CLIP_VIDEO_DEVICE,
                    audio_input=s.RETINA_CLIP_AUDIO_INPUT,
                    duration_sec=s.RETINA_CLIP_DURATION_SEC,
                    video_framerate=s.RETINA_CLIP_FRAMERATE,
                    width=s.RETINA_CLIP_WIDTH,
                    height=s.RETINA_CLIP_HEIGHT,
                    timeout_sec=s.RETINA_CLIP_TIMEOUT_SEC,
                    want_audio=want_audio,
                )
            # The audio upload is skipped entirely (not "uploaded empty") when
            # the mic was never armed: percept-store rejects a zero-byte body
            # with HTTP 400, so posting b"" would turn a deliberate, correct
            # video-only capture into a hard capture failure.
            uploads = [
                asyncio.to_thread(
                    upload_bytes,
                    result.video_bytes,
                    base_url=s.RETINA_PERCEPT_STORE_URL,
                    token=s.RETINA_PERCEPT_STORE_TOKEN or None,
                    timeout_sec=s.RETINA_PERCEPT_TIMEOUT_SEC,
                )
            ]
            if result.audio_bytes:
                uploads.append(
                    asyncio.to_thread(
                        upload_bytes,
                        result.audio_bytes,
                        base_url=s.RETINA_PERCEPT_STORE_URL,
                        token=s.RETINA_PERCEPT_STORE_TOKEN or None,
                        timeout_sec=s.RETINA_PERCEPT_TIMEOUT_SEC,
                    )
                )
            uploaded = await asyncio.gather(*uploads)
            video_sha256 = uploaded[0]
            # None, not "" -- audio_sha256 is Optional on the result payload,
            # and a caller must be able to tell "no microphone was opened"
            # from "a microphone was opened and produced something".
            audio_sha256 = uploaded[1] if len(uploaded) > 1 else None
            # Marks completion, not request time -- the cooldown above
            # measures time since a capture actually FINISHED, not since it
            # was last attempted. Set while still holding the lock so a
            # request already queued behind this one sees the up-to-date
            # value the instant it acquires the lock next.
            self._last_clip_capture_ts = time.time()
        logger.info(
            f"[RETINA] clip captured+uploaded video={video_sha256[:12]} "
            f"audio={audio_sha256[:12] if audio_sha256 else 'none(mic-not-armed)'}"
        )
        return RetinaClipCaptureResultPayload(
            ok=True,
            video_sha256=video_sha256,
            audio_sha256=audio_sha256,
            duration_sec=result.duration_sec,
            video_bytes=len(result.video_bytes),
            audio_bytes=len(result.audio_bytes) or None,
        )

    async def _clip_consume_loop(self) -> None:
        """Bus-reachable twin of POST /capture/clip -- see
        orion/bus/channels.yaml's orion:exec:request:RetinaClipCaptureService
        entry for why this exists (carbon has no reachable inbound HTTP
        surface). Mirrors orion-affectgpt-worker's _consume_loop pattern.
        """
        while not self._shutdown.is_set():
            try:
                async with self.bus.subscribe(
                    self.settings.CHANNEL_RETINA_CLIP_INTAKE
                ) as pubsub:
                    async for msg in self.bus.iter_messages(pubsub):
                        if self._shutdown.is_set():
                            break
                        data = msg.get("data")
                        if not data:
                            continue
                        decoded = self.bus.codec.decode(data)
                        if not decoded.ok or not decoded.envelope:
                            logger.error(f"[RETINA] clip RPC decode failed: {decoded.error}")
                            continue
                        asyncio.create_task(self._handle_clip_request(decoded.envelope))
            except Exception as exc:
                logger.error(f"[RETINA] clip RPC consumer error: {exc}")
                await asyncio.sleep(1)

    async def _handle_clip_request(self, envelope: BaseEnvelope) -> None:
        s = self.settings
        # Camera-identity check FIRST, before RETINA_CLIP_ENABLED or anything
        # else -- Juniper's explicit instruction, 2026-08-22: "I want this
        # to only run on my carbon webcam." This channel
        # (orion:exec:request:RetinaClipCaptureService) has no built-in
        # per-instance routing -- any retina instance subscribed to it with
        # RETINA_CLIP_ENABLED=true would otherwise respond to any request.
        # See RetinaClipCaptureRequestPayload's own docstring
        # (orion/schemas/vision.py) for why this is a structural guarantee,
        # not just an operational convention.
        try:
            payload = envelope.payload
            req = (
                payload
                if isinstance(payload, RetinaClipCaptureRequestPayload)
                else RetinaClipCaptureRequestPayload(**payload)
            )
        except Exception as exc:
            logger.error(f"[RETINA] clip request payload invalid: {exc}")
            result = RetinaClipCaptureResultPayload(
                ok=False, error=f"invalid request: {exc}", error_code="invalid_request"
            )
        else:
            if req.target_stream_id != s.RETINA_STREAM_ID:
                logger.warning(
                    f"[RETINA] clip request targeted stream_id={req.target_stream_id!r}, "
                    f"this instance is {s.RETINA_STREAM_ID!r} -- refusing"
                )
                result = RetinaClipCaptureResultPayload(
                    ok=False,
                    error=(
                        f"this instance is stream_id={s.RETINA_STREAM_ID!r}, "
                        f"request targeted {req.target_stream_id!r}"
                    ),
                    error_code="wrong_camera",
                )
            elif not s.RETINA_CLIP_ENABLED:
                result = RetinaClipCaptureResultPayload(
                    ok=False, error="RETINA_CLIP_ENABLED is false", error_code="disabled"
                )
            elif not s.RETINA_PERCEPT_STORE_URL:
                result = RetinaClipCaptureResultPayload(
                    ok=False,
                    error="RETINA_PERCEPT_STORE_URL is unset",
                    error_code="not_configured",
                )
            elif self._clip_capture_lock.locked():
                result = RetinaClipCaptureResultPayload(
                    ok=False, error="a capture is already in progress", error_code="busy"
                )
            else:
                try:
                    result = await self.capture_and_upload_clip(
                        want_audio=req.want_audio
                    )
                except ClipCaptureCooldownError as exc:
                    # Caught before the broader ClipCaptureError below --
                    # cooldown deserves its own error_code, not "capture_error".
                    logger.warning(f"[RETINA] clip capture cooldown (bus): {exc}")
                    result = RetinaClipCaptureResultPayload(
                        ok=False, error=str(exc), error_code="cooldown"
                    )
                except ClipCaptureError as exc:
                    logger.error(f"[RETINA] clip capture failed (bus): {exc}")
                    result = RetinaClipCaptureResultPayload(
                        ok=False, error=str(exc), error_code="capture_error"
                    )
                except PerceptUploadError as exc:
                    logger.error(f"[RETINA] clip upload failed (bus): {exc}")
                    result = RetinaClipCaptureResultPayload(
                        ok=False, error=str(exc), error_code="upload_error"
                    )
                except OSError as exc:
                    logger.error(f"[RETINA] clip capture could not start (bus): {exc}")
                    result = RetinaClipCaptureResultPayload(
                        ok=False,
                        error=f"capture failed to start: {exc}",
                        error_code="os_error",
                    )
                except Exception as exc:  # noqa: BLE001
                    # Unlike the HTTP route (a raised exception there just
                    # becomes a 500 the caller sees immediately), this runs
                    # in a fire-and-forget asyncio.create_task from
                    # _clip_consume_loop -- an uncaught exception here is
                    # silently swallowed by asyncio and the RPC caller gets
                    # nothing but a timeout, indistinguishable from a hung
                    # device. Always reply.
                    logger.error(f"[RETINA] clip capture failed unexpectedly (bus): {exc}")
                    result = RetinaClipCaptureResultPayload(
                        ok=False, error=str(exc), error_code="unexpected_error"
                    )

        host_ref = ServiceRef(name=s.SERVICE_NAME, version=s.SERVICE_VERSION)
        reply_envelope = envelope.derive_child(
            kind="retina.clip_capture.result", source=host_ref, payload=result
        )
        corr_id = str(envelope.correlation_id)
        reply_channel = envelope.reply_to or f"{s.CHANNEL_RETINA_CLIP_REPLY_PREFIX}:{corr_id}"
        await self.bus.publish(reply_channel, reply_envelope)

    @asynccontextmanager
    async def pause_device(self):
        """Release the physical capture device for the duration of the
        `with` block, then reopen it -- see _device_lock comment in
        __init__. Holding _device_lock for the whole duration (not just
        around the stop()/start() calls) is what actually prevents the
        race: capture_loop's own read acquires the same lock around
        capture_once() below, so it cannot be mid-read when this releases
        the device, and cannot start a new read until this has reopened it.
        """
        async with self._device_lock:
            await self.source.stop()
            try:
                yield
            finally:
                try:
                    await self.source.start()
                except Exception as exc:
                    # Don't let a failed reopen crash the caller (e.g. the
                    # HTTP handler returning a clip result) -- capture_loop's
                    # own read on its next tick will retry via source.read()'s
                    # existing "if not opened, _open()" fallback either way.
                    logger.error(f"[RETINA] failed to reopen source after pause: {exc}")

    async def capture_loop(self) -> None:
        interval = 1.0 / max(self.settings.RETINA_FPS, 0.01)
        while not self._shutdown.is_set():
            t0 = time.time()
            try:
                async with self._device_lock:
                    await self.capture_once()
            except Exception as exc:
                self.metrics.last_error = str(exc)
                logger.error(f"[RETINA] capture_once failed: {exc}")
            if time.time() - self._last_cleanup > 10:
                await asyncio.to_thread(
                    cleanup_old_frames,
                    self.settings.FRAME_STORAGE_DIR,
                    self.settings.FRAME_RETENTION_SECONDS,
                )
                self._last_cleanup = time.time()
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def capture_once(self) -> bool:
        self._sample_attempted = True
        result = await self.source.read()
        if result is None:
            self.metrics.frames_failed += 1
            self.metrics.last_error = "source read returned no frame"
            return False
        if str(self.settings.RETINA_FRAME_MODE).strip().lower() == "percept_store":
            if not self.settings.RETINA_PERCEPT_STORE_URL:
                self.metrics.frames_failed += 1
                self.metrics.last_error = (
                    "RETINA_FRAME_MODE=percept_store but RETINA_PERCEPT_STORE_URL is unset"
                )
                logger.error(f"[RETINA] {self.metrics.last_error}")
                return False
            try:
                saved = await asyncio.to_thread(
                    upload_frame,
                    result.frame,
                    base_url=self.settings.RETINA_PERCEPT_STORE_URL,
                    quality=self.settings.JPEG_QUALITY,
                    token=self.settings.RETINA_PERCEPT_STORE_TOKEN or None,
                    timeout_sec=self.settings.RETINA_PERCEPT_TIMEOUT_SEC,
                )
            except PerceptUploadError as exc:
                # Drop the frame and try the next one. Deliberately no spooling:
                # this mode runs on laptops that sleep and roam, and a backlog
                # of webcam images on a personal machine is a worse failure than
                # a gap in the record.
                self.metrics.frames_failed += 1
                self.metrics.last_error = str(exc)
                logger.warning(f"[RETINA] {exc}")
                return False
        else:
            saved = await asyncio.to_thread(
                save_frame,
                result.frame,
                directory=self.settings.FRAME_STORAGE_DIR,
                camera_id=self.settings.RETINA_CAMERA_ID,
                stream_id=self.settings.RETINA_STREAM_ID,
                ts=result.ts,
                quality=self.settings.JPEG_QUALITY,
            )
        payload = VisionFramePointerPayload(
            image_path=saved.image_path or None,
            sha256=saved.sha256,
            camera_id=self.settings.RETINA_CAMERA_ID,
            stream_id=self.settings.RETINA_STREAM_ID,
            frame_ts=result.ts,
            width=saved.width,
            height=saved.height,
            format=saved.format,
        )
        env = make_frame_pointer_envelope(
            payload,
            service_name=self.settings.SERVICE_NAME,
            service_version=self.settings.SERVICE_VERSION,
        )
        await self.bus.publish(self.settings.CHANNEL_RETINA_PUB, env)
        now = time.time()
        self.metrics.frames_published += 1
        self.metrics.last_frame_ts = result.ts
        self.metrics.last_error = None
        if self._last_publish_ts is not None:
            dt = now - self._last_publish_ts
            if dt > 0:
                self.metrics.fps_observed = 1.0 / dt
        elif self.metrics.frames_published == 1:
            elapsed = now - self._started_at
            if elapsed > 0:
                self.metrics.fps_observed = 1.0 / elapsed
        self._last_publish_ts = now
        logger.info(f"[RETINA] Published frame pointer: {saved.image_path}")
        return True

    def _source_ok(self) -> bool:
        if not self._sample_attempted:
            return True
        if self.metrics.last_error is not None:
            return False
        if self.metrics.frames_published > 0:
            return True
        return self.metrics.frames_failed == 0

    async def _health_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                env = make_system_health_envelope(
                    service_name=self.settings.SERVICE_NAME,
                    service_version=self.settings.SERVICE_VERSION,
                    camera_id=self.settings.RETINA_CAMERA_ID,
                    stream_id=self.settings.RETINA_STREAM_ID,
                    source_type=self.settings.RETINA_SOURCE_TYPE,
                    source_ok=self._source_ok(),
                    metrics=self.metrics,
                    fps_target=self.settings.RETINA_FPS,
                    storage_dir=self.settings.FRAME_STORAGE_DIR,
                )
                await self.bus.publish(self.settings.CHANNEL_SYSTEM_HEALTH, env)
            except Exception as exc:
                logger.warning(f"[RETINA] health publish failed: {exc}")
            await asyncio.sleep(self.settings.HEALTH_INTERVAL_SECONDS)


service = RetinaService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()


app = FastAPI(title="Orion Vision Retina", version="0.2.0", lifespan=lifespan)


@app.post("/capture/clip")
async def capture_clip_endpoint(request: Request):
    """On-demand video+audio clip capture for AffectGPT. Live-verified
    against real hardware, 2026-08-22 -- see app/clip_capture.py module
    docstring for what that run found and fixed.

    HTTP path for an operator on the same host/tailnet as this port (carbon's
    venv/systemd deploy binds 127.0.0.1 only). Callers with no network path
    to this node at all (e.g. Hub, via orion-juniper-affective-state) use
    the bus RPC twin instead -- see RetinaService._handle_clip_request /
    orion:exec:request:RetinaClipCaptureService in orion/bus/channels.yaml.
    Both share RetinaService.capture_and_upload_clip() so there is exactly
    one capture implementation, not two that can drift.

    A caller (curl, or Hub via the bus) gets back sha256 refs. **Those refs
    are not yet consumable by the worker directly** -- orion-affectgpt-worker
    still requires local video_path/audio_path; orion-juniper-affective-state
    does the percept-store fetch-by-hash + temp-file bridge on circe (see
    that service's app/main.py capture_and_assess()).

    Requires ``?target_stream_id=<this instance's RETINA_STREAM_ID>`` in the
    query string -- review finding, 2026-08-22: the bus RPC twin
    (_handle_clip_request) got a required target_stream_id check the same
    day this route did NOT, so the camera-identity guarantee was fully
    bypassable via a plain curl even though this module's own docstring and
    README claimed it "holds even if a second instance is misconfigured."
    Now both entry points enforce it the same way.
    """
    s = service.settings
    target_stream_id = request.query_params.get("target_stream_id")
    if target_stream_id != s.RETINA_STREAM_ID:
        logger.warning(
            f"[RETINA] HTTP clip request targeted stream_id={target_stream_id!r}, "
            f"this instance is {s.RETINA_STREAM_ID!r} -- refusing"
        )
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    f"this instance is stream_id={s.RETINA_STREAM_ID!r}, "
                    f"request targeted {target_stream_id!r}"
                ),
                "error_code": "wrong_camera",
            },
            status_code=400,
        )
    if not s.RETINA_CLIP_ENABLED:
        return JSONResponse(
            {"ok": False, "error": "RETINA_CLIP_ENABLED is false"}, status_code=503
        )
    if not s.RETINA_PERCEPT_STORE_URL:
        return JSONResponse(
            {"ok": False, "error": "RETINA_PERCEPT_STORE_URL is unset"}, status_code=503
        )
    # Shared-secret gate -- see settings.py RETINA_CLIP_TOKEN comment for why
    # this exists (this endpoint triggers a live recording, unlike every
    # other route on this service, which is bus-only or read-only).
    if s.RETINA_CLIP_TOKEN:
        if request.headers.get("X-Orion-Retina-Token") != s.RETINA_CLIP_TOKEN:
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    if service._clip_capture_lock.locked():
        return JSONResponse(
            {"ok": False, "error": "a capture is already in progress"}, status_code=409
        )

    try:
        result = await service.capture_and_upload_clip()
    except ClipCaptureCooldownError as exc:
        # Caught before the broader ClipCaptureError below.
        logger.warning(f"[RETINA] clip capture cooldown: {exc}")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=429)
    except ClipCaptureError as exc:
        logger.error(f"[RETINA] clip capture failed: {exc}")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    except OSError as exc:
        # e.g. RETINA_CLIP_FFMPEG_BIN misconfigured / ffmpeg not
        # installed -- create_subprocess_exec raises this directly,
        # unguarded, before clip_capture.py gets a chance to wrap it
        # (review finding, 2026-08-22: this used to be an unhandled 500).
        logger.error(f"[RETINA] clip capture could not start: {exc}")
        return JSONResponse({"ok": False, "error": f"capture failed to start: {exc}"}, status_code=502)
    except PerceptUploadError as exc:
        logger.error(f"[RETINA] clip upload failed: {exc}")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    except Exception as exc:  # noqa: BLE001
        # Symmetry with _handle_clip_request's own catch-all (review
        # finding, 2026-08-22): an exception type outside the three above
        # used to propagate uncaught here, returning a bare unstructured 500
        # instead of the {"ok": false, "error": ...} shape every other
        # branch (and this docstring) promises.
        logger.error(f"[RETINA] clip capture failed unexpectedly: {exc}")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    return result.model_dump(exclude_none=True)
