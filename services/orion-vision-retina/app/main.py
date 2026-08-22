from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.vision import VisionFramePointerPayload

from .clip_capture import ClipCaptureError, capture_clip
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

    async def start(self) -> None:
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=self.settings.LOG_LEVEL)

        await self.bus.connect()
        await self.source.start()
        self._shutdown.clear()
        self._capture_task = asyncio.create_task(self.capture_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info(f"[RETINA] Started → {self.settings.CHANNEL_RETINA_PUB}")

    async def stop(self) -> None:
        self._shutdown.set()
        for task in (self._capture_task, self._health_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.source.stop()
        await self.bus.close()

    async def capture_loop(self) -> None:
        interval = 1.0 / max(self.settings.RETINA_FPS, 0.01)
        while not self._shutdown.is_set():
            t0 = time.time()
            try:
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

# One capture at a time: /dev/video0 is exclusive-access, and a second ffmpeg
# trying to open it mid-capture fails with an opaque device-busy error
# instead of a clear "already in progress" response (review finding,
# 2026-08-22).
_clip_capture_lock = asyncio.Lock()


@app.post("/capture/clip")
async def capture_clip_endpoint(request: Request):
    """On-demand video+audio clip capture for AffectGPT. See
    app/clip_capture.py module docstring: UNVERIFIED against real hardware.

    No toggle, no bus wiring yet -- deliberately on-demand only for this
    first patch (see services/orion-juniper-affective-state/README.md non-
    goals). A caller (curl, or later Hub) hits this and gets back sha256
    refs. **Those refs are not yet consumable end-to-end**:
    orion-affectgpt-worker currently requires local video_path/audio_path,
    not a percept-store ref -- fetch-by-hash on that side is real, separate,
    not-yet-built follow-up work (review finding, 2026-08-22: this docstring
    previously implied more readiness than exists).
    """
    s = service.settings
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

    if _clip_capture_lock.locked():
        return JSONResponse(
            {"ok": False, "error": "a capture is already in progress"}, status_code=409
        )

    async with _clip_capture_lock:
        try:
            result = await capture_clip(
                ffmpeg_bin=s.RETINA_CLIP_FFMPEG_BIN,
                video_device=s.RETINA_CLIP_VIDEO_DEVICE,
                audio_input=s.RETINA_CLIP_AUDIO_INPUT,
                duration_sec=s.RETINA_CLIP_DURATION_SEC,
                video_framerate=s.RETINA_CLIP_FRAMERATE,
                width=s.RETINA_CLIP_WIDTH,
                height=s.RETINA_CLIP_HEIGHT,
                timeout_sec=s.RETINA_CLIP_TIMEOUT_SEC,
            )
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

        try:
            video_sha256, audio_sha256 = await asyncio.gather(
                asyncio.to_thread(
                    upload_bytes,
                    result.video_bytes,
                    base_url=s.RETINA_PERCEPT_STORE_URL,
                    token=s.RETINA_PERCEPT_STORE_TOKEN or None,
                    timeout_sec=s.RETINA_PERCEPT_TIMEOUT_SEC,
                ),
                asyncio.to_thread(
                    upload_bytes,
                    result.audio_bytes,
                    base_url=s.RETINA_PERCEPT_STORE_URL,
                    token=s.RETINA_PERCEPT_STORE_TOKEN or None,
                    timeout_sec=s.RETINA_PERCEPT_TIMEOUT_SEC,
                ),
            )
        except PerceptUploadError as exc:
            logger.error(f"[RETINA] clip upload failed: {exc}")
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)

    logger.info(
        f"[RETINA] clip captured+uploaded video={video_sha256[:12]} audio={audio_sha256[:12]}"
    )
    return {
        "ok": True,
        "video_sha256": video_sha256,
        "audio_sha256": audio_sha256,
        "duration_sec": result.duration_sec,
        "video_bytes": len(result.video_bytes),
        "audio_bytes": len(result.audio_bytes),
    }
