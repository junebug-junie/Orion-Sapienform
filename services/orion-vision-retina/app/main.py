from __future__ import annotations

import asyncio
import os
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


# Longest self-look key value (a handoff id) kept for logs and health.
SELF_LOOK_VALUE_MAX_CHARS = 80


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
            rtsp_transport=self.settings.RETINA_RTSP_TRANSPORT,
            rtsp_stall_seconds=self.settings.RETINA_RTSP_STALL_SECONDS,
            rtsp_io_timeout_seconds=self.settings.RETINA_RTSP_IO_TIMEOUT_SECONDS,
            rtsp_reconnect_quiet_seconds=self.settings.RETINA_RTSP_RECONNECT_QUIET_SECONDS,
            rtsp_session_state_path=self.settings.rtsp_session_state_path(),
            rtsp_heal_enabled=self.settings.RETINA_RTSP_HEAL_ENABLED,
            rtsp_heal_check_every=self.settings.RETINA_RTSP_HEAL_CHECK_EVERY,
            rtsp_heal_window_seconds=self.settings.RETINA_RTSP_HEAL_WINDOW_SECONDS,
            rtsp_heal_min_share=self.settings.RETINA_RTSP_HEAL_MIN_SHARE,
            ring_state_dir=self.settings.RETINA_RING_STATE_DIR,
            ring_burst_seconds=self.settings.RETINA_RING_BURST_SECONDS,
            ring_cooldown_seconds=self.settings.RETINA_RING_COOLDOWN_SECONDS,
            ring_event_kinds=self.settings.ring_event_kinds(),
            ring_self_look_max_per_day=self.settings.RETINA_RING_SELF_LOOK_MAX_PER_DAY,
            ring_self_look_min_spacing_seconds=self.settings.RETINA_RING_SELF_LOOK_MIN_SPACING_SECONDS,
        )
        # Ring source: every recorded push is published as events.
        if hasattr(self.source, "event_sink"):
            self.source.event_sink = self._on_ring_push
        self._ring_publish_tasks: set[asyncio.Task] = set()
        self._self_look_task: Optional[asyncio.Task] = None
        # Last key value acted on: one live view per handoff, not one per poll.
        self._self_look_last_value: str | None = None
        self._self_look_last_logged: tuple[str, str] | None = None
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
        # Publisher watchdog (RETINA_PUBLISH_STALL_SECONDS). The stall clock
        # restarts whenever "no frame" is the expected state (RTSP quiet
        # wait, Ring between events), so those never count as a stall.
        self._stall_clock_ts: float = self._started_at
        # Monotonic time capture_loop last began an iteration: frozen = the
        # loop is hung or gone, whatever the source is doing. Monotonic so an
        # NTP step or a laptop resume (carbon) does not read as a hang.
        self._loop_tick_mono: float = time.monotonic()
        # Monotonic time a frame was read but could not be published (save,
        # upload or bus error). A restart does not fix those, so the
        # "streaming but not publishing" exit stands down while they recur.
        self._last_capture_error_mono: float | None = None
        # Injectable for tests. os._exit, not sys.exit: this runs inside the
        # event loop, and the point is that the process really ends so the
        # container restart policy brings the eye back.
        self._exit = os._exit
        self._exiting = False
        self._exit_started = False
        self._exit_task: Optional[asyncio.Task] = None
        self._stall_logged: str | None = None

    async def start(self) -> None:
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=self.settings.LOG_LEVEL)

        await self.bus.connect()
        await self.source.start()
        self._shutdown.clear()
        # The boot grace starts now, not at module import.
        self._started_at = self._stall_clock_ts = time.time()
        self._loop_tick_mono = time.monotonic()
        self._capture_task = self._spawn_forever(self.capture_loop(), "capture_loop")
        self._health_task = self._spawn_forever(self._health_loop(), "health_loop")
        if self.settings.RETINA_RING_SELF_LOOK_ENABLED and hasattr(self.source, "self_look"):
            self._self_look_task = self._spawn_forever(self._self_look_loop(), "self_look_loop")
            logger.info(f"[RETINA] ring self-look watching {self.settings.RETINA_RING_SELF_LOOK_KEY}")
        if self.settings.RETINA_CLIP_ENABLED:
            self._clip_consumer_task = self._spawn_forever(self._clip_consume_loop(), "clip_consume_loop")
            logger.info(
                f"[RETINA] clip RPC intake → {self.settings.CHANNEL_RETINA_CLIP_INTAKE}"
            )
        logger.info(f"[RETINA] Started → {self.settings.CHANNEL_RETINA_PUB}")

    def _spawn_forever(self, coro, name: str) -> asyncio.Task:
        """Start a task that must run until stop(). If it ends any other way,
        exit the process.

        2026-10-17: capture_loop died on an exception and nothing logged it
        (the service held a reference to the task, so asyncio's "exception
        was never retrieved" never fired). Health kept saying ok for 3 h.
        A dead eye must be loud and must come back: CRITICAL log, then exit
        so the container restart policy restarts it.
        """
        task = asyncio.create_task(coro, name=f"retina-{name}")
        task.add_done_callback(lambda t: self._on_forever_task_done(t, name))
        return task

    def _on_forever_task_done(self, task: asyncio.Task, name: str) -> None:
        if self._shutdown.is_set() or self._exiting:
            return  # normal stop(), or already exiting
        if task.cancelled():
            reason = "was cancelled"
        else:
            exc = task.exception()
            reason = f"died: {type(exc).__name__}: {exc}" if exc else "returned"
            if exc is not None:
                logger.opt(exception=exc).critical(f"[RETINA] {name} traceback")
        self._die_soon(f"{name} {reason}; it must run forever")

    def _die_soon(self, reason: str) -> None:
        """Sync entry point (done callbacks): start the exit as a task."""
        if self._exiting:
            return
        self._exiting = True  # set now: a second callback must not schedule another
        self._exit_task = asyncio.get_running_loop().create_task(self._die(reason))

    async def _die(self, reason: str) -> None:
        """Log CRITICAL, release the camera, exit once. The container restart
        policy (and the RTSP quiet period on the way back up) does the rest.

        source.stop() first, bounded: an RTSP reader that releases its
        capture sends TEARDOWN, so the camera does not keep an orphaned
        session (the 2026-10-16 smear hazard). A wedged reader cannot, and
        the bound keeps it from blocking the exit."""
        if self._exit_started:
            return
        self._exiting = True
        self._exit_started = True
        logger.critical(f"[RETINA] {reason}. Exiting for restart.")
        timeout = min(30.0, float(getattr(self.source, "_join_timeout", 10.0) or 10.0) + 2.0)
        try:
            await asyncio.wait_for(self.source.stop(), timeout=timeout)
        except BaseException as exc:  # noqa: BLE001 -- nothing may stop the exit
            logger.error(f"[RETINA] source stop before exit failed: {type(exc).__name__}: {exc}")
        self._exit(1)

    async def stop(self) -> None:
        self._shutdown.set()
        for task in (self._capture_task, self._health_task, self._clip_consumer_task, self._self_look_task):
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
            self._loop_tick_mono = time.monotonic()
            try:
                async with self._device_lock:
                    await self.capture_once()
            except Exception as exc:
                self._last_capture_error_mono = time.monotonic()
                self.metrics.last_error = str(exc)
                logger.error(f"[RETINA] capture_once failed: {type(exc).__name__}: {exc}")
            # Housekeeping must never stop capture: the 2026-10-17 outage was
            # this call raising outside any try and killing the loop.
            try:
                if time.time() - self._last_cleanup > 10:
                    self._last_cleanup = time.time()
                    await asyncio.to_thread(
                        cleanup_old_frames,
                        self.settings.FRAME_STORAGE_DIR,
                        self.settings.FRAME_RETENTION_SECONDS,
                        camera_id=self.settings.RETINA_CAMERA_ID,
                        stream_id=self.settings.RETINA_STREAM_ID,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[RETINA] frame cleanup failed (capture continues): {type(exc).__name__}: {exc}")
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    def _on_ring_push(self, push) -> None:  # noqa: ANN001 -- ring_source.RingPush
        """RingEventFrameSource.event_sink (sync, on the loop): publish without
        blocking the push listener."""
        task = asyncio.get_running_loop().create_task(self._publish_ring_push(push))
        self._ring_publish_tasks.add(task)
        task.add_done_callback(self._ring_publish_tasks.discard)

    async def _publish_ring_push(self, push) -> None:  # noqa: ANN001
        from .ring_events import build_street_event, build_vision_event, street_event_envelope, vision_event_envelope

        s = self.settings
        sid = s.RETINA_STREAM_ID
        item = build_vision_event(push, stream_id=sid)
        if item is not None:
            try:
                env = vision_event_envelope(push, item, service_name=s.SERVICE_NAME, service_version=s.SERVICE_VERSION)
                await self.bus.publish(s.CHANNEL_VISION_EVENTS_PUB, env)
                self.metrics.ring_vision_events_published += 1
                logger.info(
                    f"[RETINA] ring vision_event_published type={item.event_type} event_id={item.event_id} "
                    f"corr={push.correlation_id} channel={s.CHANNEL_VISION_EVENTS_PUB}"
                )
            except Exception as exc:  # noqa: BLE001
                self.metrics.ring_events_publish_failed += 1
                logger.warning(f"[RETINA] ring vision_event_publish_failed corr={push.correlation_id}: {exc}")
        street = build_street_event(push, stream_id=sid)
        if street is not None:
            try:
                env = street_event_envelope(push, street, service_name=s.SERVICE_NAME, service_version=s.SERVICE_VERSION)
                await self.bus.publish(s.CHANNEL_STREET_EVENTS_PUB, env)
                self.metrics.ring_street_events_published += 1
                logger.info(
                    f"[RETINA] ring street_event_published kind={street.kind} event_id={street.event_id} "
                    f"corr={push.correlation_id} channel={s.CHANNEL_STREET_EVENTS_PUB}"
                )
            except Exception as exc:  # noqa: BLE001
                self.metrics.ring_events_publish_failed += 1
                logger.warning(f"[RETINA] ring street_event_publish_failed corr={push.correlation_id}: {exc}")

    async def self_look_once(self) -> str | None:
        """One poll of the self-look key. Returns the outcome, or None when
        the key is absent or already acted on."""
        raw = await self.bus.redis.get(self.settings.RETINA_RING_SELF_LOOK_KEY)
        if raw is None:
            self._self_look_last_value = None  # key expired: the next SET counts
            return None
        value = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)).strip() or "unnamed"
        value = value[:SELF_LOOK_VALUE_MAX_CHARS]  # set by another service: bound what we log
        if value == self._self_look_last_value:
            return None
        outcome = self.source.self_look(value)
        # started / an open burst already looking / today's cap: done with
        # this value. cooldown / spacing / not_ready: retry while the key lives.
        if outcome in ("started", "burst_running", "daily_cap"):
            self._self_look_last_value = value
        if (value, outcome) != self._self_look_last_logged:
            self._self_look_last_logged = (value, outcome)
            logger.info(
                f"[RETINA] ring self_look handoff={value!r} outcome={outcome} "
                f"today={self.source.self_looks_today}/{self.settings.RETINA_RING_SELF_LOOK_MAX_PER_DAY}"
            )
        return outcome

    async def _self_look_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                await self.self_look_once()
            except Exception as exc:  # noqa: BLE001 -- a bus hiccup must not kill the loop
                logger.warning(f"[RETINA] ring self_look poll failed: {type(exc).__name__}: {exc}")
            await asyncio.sleep(self.settings.RETINA_RING_SELF_LOOK_POLL_SECONDS)

    async def capture_once(self) -> bool:
        self._sample_attempted = True
        result = await self.source.read()
        if getattr(self.source, "needs_process_restart", False) is True:
            # RTSP readers wedged in FFmpeg and the session cap stops us opening
            # more: only a fresh process clears them. Exit so the container's
            # restart policy brings the eye back, instead of a silent outage.
            await self._die(str(getattr(self.source, "stall_error", "") or "rtsp readers wedged"))
            return False
        self._copy_source_metrics()
        if result is None:
            if getattr(self.source, "waiting_for_event", False) is True:
                # Event camera (Ring) between events: no frame is the normal
                # state. A failed live view sets the source's last_error.
                self.metrics.last_error = getattr(self.source, "last_error", None)
                return False
            quiet = self.metrics.source_quiet_wait_remaining_sec
            if quiet > 0:
                # Deliberate: waiting out the camera's stale session before
                # reconnecting. Not a failed read.
                self.metrics.last_error = f"rtsp quiet period: {quiet:.0f}s before connecting to the camera"
                return False
            if getattr(self.source, "ever_connected", True) is False:
                # A live RTSP source still connecting after boot/reconnect:
                # not a failure yet.
                self.metrics.last_error = getattr(self.source, "last_error", None) or "connecting to camera"
                return False
            self.metrics.frames_failed += 1
            # An RTSP stall names itself ("rtsp stalled: no frame for 47s").
            self.metrics.last_error = getattr(self.source, "stall_error", None) or "source read returned no frame"
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
                self._last_capture_error_mono = time.monotonic()
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
        ring_corr = (result.source_meta or {}).get("ring_correlation_id")
        suffix = f" ring_corr={ring_corr}" if ring_corr else ""
        logger.info(f"[RETINA] Published frame pointer: {saved.image_path}{suffix}")
        return True

    def _copy_source_metrics(self) -> None:
        src = self.source
        m = self.metrics
        m.source_stall_restarts = int(getattr(src, "stall_restarts", 0) or 0)
        if hasattr(src, "events_seen"):  # Ring
            m.ring_events_seen = int(src.events_seen)
            m.ring_events_ignored = int(src.events_ignored)
            m.ring_bursts_started = int(src.bursts_started)
            m.ring_bursts_failed = int(src.bursts_failed)
            m.ring_last_event_kind = src.last_event_kind
            m.ring_last_event_ts = src.last_event_ts
            m.ring_dings_total = int(src.dings_total)
            m.ring_motions_total = int(src.motions_total)
            m.ring_last_ding_ts = src.last_ding_ts
            m.ring_bursts_extended = int(src.bursts_extended)
            m.ring_ding_reopens_refused = int(src.ding_reopens_refused)
            m.ring_self_looks_today = int(src.self_looks_today)
            m.ring_self_looks_total = int(src.self_looks_total)
            m.ring_last_self_look_ts = src.last_self_look_ts
            m.ring_last_self_look_reason = src.last_self_look_reason
        if not hasattr(src, "quiet_wait_remaining"):
            return  # not RTSP
        m.source_quiet_wait_remaining_sec = float(src.quiet_wait_remaining())
        m.source_last_session_end_ts = src.last_session_end_ts
        m.source_corrupt_frame_rate = src.corrupt_frame_rate()
        m.source_corruption_checks = int(src.corruption_checks)
        m.source_corrupt_frames = int(src.corrupt_frames)
        m.source_corruption_heals = int(src.corruption_heals)
        m.source_corruption_heal_suppressed = bool(getattr(src, "heal_suppressed", False))

    def _source_ok(self) -> bool:
        if not self._sample_attempted:
            return True
        if self.metrics.last_error is not None:
            return False
        if self.metrics.frames_published > 0:
            return True
        return self.metrics.frames_failed == 0

    def _expected_quiet(self) -> bool:
        """No frame is the normal state right now: RTSP waiting out the
        camera's stale session, or an event camera (Ring) between events.
        Read live from the source, not from metrics (capture_loop copies
        those, and capture_loop may be the thing that is stuck)."""
        src = self.source
        if getattr(src, "waiting_for_event", False) is True:
            return True
        qwr = getattr(src, "quiet_wait_remaining", None)
        if callable(qwr):
            try:
                return float(qwr()) > 0
            except Exception:  # noqa: BLE001
                return False
        return False

    def publish_stall(self, now: float | None = None) -> str | None:
        """Why nothing has been published for RETINA_PUBLISH_STALL_SECONDS,
        or None. Health reports it as `capture_stalled`, status degraded."""
        limit = float(self.settings.RETINA_PUBLISH_STALL_SECONDS)
        if limit <= 0:
            return None
        now = time.time() if now is None else now
        if self._expected_quiet():
            self._stall_clock_ts = now
            return None
        since = max(self._last_publish_ts or 0.0, self._stall_clock_ts, self._started_at)
        age = now - since
        if age <= limit:
            return None
        return f"no frame published for {age:.0f}s (limit {limit:.0f}s)"

    def publisher_dead(self, now: float | None = None, mono: float | None = None) -> str | None:
        """A reason to exit for restart, or None.

        Only cases a restart can fix and that are certain:
          - capture_loop has not started an iteration for the limit (hung or
            gone), whatever the source is doing;
          - the source is delivering frames (RTSP reader clock, independent of
            capture_loop) but nothing was published for the limit, and no
            capture error (save/upload/bus) happened in that window.
        A camera that is down, or a disk/bus/upload failure, is not this: it
        reads degraded with last_error, no exit (a restart would not fix it
        and every exit costs the camera a session).
        """
        limit = float(self.settings.RETINA_PUBLISH_STALL_SECONDS)
        if limit <= 0:
            return None
        now = time.time() if now is None else now
        mono = time.monotonic() if mono is None else mono
        loop_age = mono - self._loop_tick_mono
        # A paused device (clip capture) holds the lock for at most the clip
        # timeout; the limit is validated to exceed one capture interval.
        if loop_age > limit:
            return f"capture_loop has not ticked for {loop_age:.0f}s"
        stall = self.publish_stall(now)
        if stall is None:
            return None
        err = self._last_capture_error_mono
        if err is not None and mono - err <= limit:
            return None
        src_age_fn = getattr(self.source, "last_frame_age", None)
        if not callable(src_age_fn):
            return None
        try:
            src_age = src_age_fn()
        except Exception:  # noqa: BLE001
            return None
        if src_age is not None and src_age < limit:
            return f"source is streaming (newest frame {src_age:.0f}s old) but {stall}"
        return None

    def health_envelope(self, stall: str | None) -> BaseEnvelope:
        return make_system_health_envelope(
            service_name=self.settings.SERVICE_NAME,
            service_version=self.settings.SERVICE_VERSION,
            camera_id=self.settings.RETINA_CAMERA_ID,
            stream_id=self.settings.RETINA_STREAM_ID,
            source_type=self.settings.RETINA_SOURCE_TYPE,
            source_ok=self._source_ok(),
            metrics=self.metrics,
            capture_stalled=stall,
            heartbeat_interval_sec=self.settings.HEALTH_INTERVAL_SECONDS,
            fps_target=self.settings.RETINA_FPS,
            storage_dir=self.settings.FRAME_STORAGE_DIR,
        )

    async def _health_loop(self) -> None:
        while not self._shutdown.is_set():
            stall: str | None = None
            try:
                stall = self.publish_stall()
                env = self.health_envelope(stall)
                await self.bus.publish(self.settings.CHANNEL_SYSTEM_HEALTH, env)
            except Exception as exc:
                logger.warning(f"[RETINA] health publish failed: {exc}")
            if stall is not None and self._stall_logged is None:
                logger.warning(f"[RETINA] capture stalled: {stall}")
            elif stall is None and self._stall_logged is not None:
                logger.info("[RETINA] capture publishing again")
            self._stall_logged = stall
            dead = self.publisher_dead()
            if dead is not None:
                await self._die(f"publisher watchdog: {dead}")
                return
            await asyncio.sleep(self.settings.HEALTH_INTERVAL_SECONDS)


service = RetinaService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()


app = FastAPI(title="Orion Vision Retina", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health_endpoint():
    """The same SystemHealthV1 payload the health loop publishes, read now."""
    service._copy_source_metrics()
    return service.health_envelope(service.publish_stall()).payload


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
