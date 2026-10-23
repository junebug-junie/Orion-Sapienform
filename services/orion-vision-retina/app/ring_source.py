"""Event-driven frames from a Ring doorbell (RETINA_SOURCE_TYPE=ring).

A battery Ring doorbell with no Ring Protect plan has no RTSP, no saved
recordings, and cannot take a snapshot while it records a motion event
(https://github.com/dgreif/ring/wiki/Snapshot-Limitations). The one picture
source left is a live view. So: listen for Ring's push events (motion, ding);
on each one, open a short WebRTC live view, keep the newest decoded frame,
and hang up after RETINA_RING_BURST_SECONDS. RetinaService.capture_loop
samples that frame at RETINA_FPS exactly as it samples an RTSP camera, so
downstream sees the same frame-pointer stream as the walkway, only in bursts.

Never continuous: no live view runs without an event (or a capped self-look,
`self_look`), one burst at a time, and RETINA_RING_COOLDOWN_SECONDS between
the end of a burst and the next start (each live view wakes the camera and
costs battery). A ding is the exception: every push is recorded (event_sink),
and a ding extends an open burst or ignores the cooldown. Every exit path of
a burst closes the Ring session: an orphaned one keeps the camera streaming
to nobody.

The push link is supervised: firebase-messaging stops for good after a few
connection errors, and ring_doorbell refreshes its session only once. A
listener that stopped is reported (``ring push listener stopped``) and
rebuilt, and the whole connection is rebuilt every RECONNECT_EVERY_SECONDS,
so a deaf doorbell never reads as a quiet one.

Auth: run ``python -m app.ring_login`` once inside the container. It writes
the refresh token under RETINA_RING_STATE_DIR; this source keeps it fresh.
Without a token the source reports ``ring not authenticated`` and checks
again every SUPERVISE_SECONDS, so no restart is needed after login.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from .sources import FrameReadResult

USER_AGENT = "orion-vision-retina"
TOKEN_FILE = "ring_token.json"
FCM_FILE = "ring_fcm_credentials.json"
HARDWARE_ID_FILE = "ring_hardware_id"
# Ring push kinds that mean "someone or something is at the door".
DEFAULT_EVENT_KINDS = ("motion", "ding")
SUPERVISE_SECONDS = 60.0
# ring_doorbell refreshes the Ring session once, at 12 h, then never again.
RECONNECT_EVERY_SECONDS = 12 * 3600.0
# Seconds to wait for Ring's SDP answer to our live-view offer.
ANSWER_TIMEOUT_SECONDS = 10.0
# aiortc buffers at most 128 RTP packets per video stream. A frame bigger
# than that loses its head to make room, and the tail goes to the decoder,
# which cannot decode it. Since ~2026-10-21 Ring sends this doorbell's
# 1920x1920 H.264 with keyframes of ~250 packets, so no keyframe ever
# decoded and every live view ended "no frame within 15s" while ~7,000 good
# video packets arrived (measured 2026-10-23; 512 decoded 261 frames in 12 s).
# 1024 gives 4x headroom; must be a power of 2 (aiortc asserts it).
VIDEO_JITTER_CAPACITY = 1024
# This many live views in a row with no frame = the doorbell is blind.
BLIND_AFTER_FAILED_BURSTS = 3


def ensure_video_jitter_capacity(video_capacity: int = VIDEO_JITTER_CAPACITY) -> None:
    """Make aiortc's video receivers buffer ``video_capacity`` packets (idempotent).

    aiortc hard-codes ``JitterBuffer(capacity=128, is_video=True)`` in
    RTCRtpReceiver; swap the class it looks up for one that raises a video
    capacity. Audio is untouched."""
    try:
        import aiortc.jitterbuffer as jb
        import aiortc.rtcrtpreceiver as rr
    except ImportError as exc:  # an aiortc without these modules (or a test double)
        logger.warning(f"[RETINA] ring video buffer not raised: {exc}")
        return

    current = rr.JitterBuffer
    if getattr(current, "_orion_video_capacity", None) == video_capacity:
        return
    base = getattr(current, "_orion_base", current)

    class _VideoJitterBuffer(base):  # type: ignore[misc,valid-type]
        _orion_video_capacity = video_capacity
        _orion_base = base

        # Same parameter names as aiortc: RTCRtpReceiver passes them by keyword.
        def __init__(self, capacity: int, prefetch: int = 0, is_video: bool = False) -> None:
            if is_video:
                capacity = max(capacity, video_capacity)
            super().__init__(capacity=capacity, prefetch=prefetch, is_video=is_video)

    assert base is jb.JitterBuffer or issubclass(base, jb.JitterBuffer)
    rr.JitterBuffer = _VideoJitterBuffer
# A ring during an open live view extends it, up to this many burst lengths
# in total (a stuck button must not stream all day).
MAX_BURST_FACTOR = 4.0
# A ding skips the cooldown, so a stuck button or repeated presses could keep
# the battery camera streaming. At most this many ding-opened live views that
# skipped the cooldown per window; later ones are still recorded.
DING_REOPEN_MAX = 3
DING_REOPEN_WINDOW_SECONDS = 600.0

# Every spelling ring_doorbell 0.9.14 can hand us for a press or a motion.
# Legacy (gcmData) pushes: listen/eventlistener.py _get_ding_event compares
# `action.lower == ...` (the method, never called), so a legacy ding never
# matches and its kind is the raw action string "com.ring.push.HANDLE_NEW_DING".
# New-format pushes map the category through PUSH_NOTIFICATION_KINDS; an
# unmapped category reads "Unknown" (logged, not counted).
KIND_ALIASES = {
    "ding": "ding",
    "com.ring.push.handle_new_ding": "ding",
    "com.ring.pn.live-event.ding": "ding",
    "motion": "motion",
    "com.ring.push.handle_new_motion": "motion",
    "com.ring.pn.live-event.motion": "motion",
}
SELF_LOOK_KIND = "self_look"


def normalize_kind(raw: Any) -> str:
    k = str(raw or "").strip().lower()
    return KIND_ALIASES.get(k, k)


@dataclass(frozen=True)
class RingPush:
    """One recorded Ring push (a configured kind on our doorbell)."""

    kind: str  # normalized: "ding" | "motion" | another configured kind
    raw_kind: str
    ring_event_id: str | None
    occurred_at: float  # Ring's own created_at when present, else receipt
    received_ts: float
    correlation_id: str
    # started | extended | queued (opens when the live view connects, or
    # reopens after it closes) | already_open | cooldown | ding_capped
    live_view: str


def _write_private(path: Path, data: str) -> None:
    """Atomic, owner-only write: these files are Ring account credentials.
    A unique temp name, so two writers (login + a token refresh) never
    interleave into one file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def hardware_id(state_dir: str) -> str:
    """Stable per-install id. ring_doorbell derives one from the MAC address,
    which changes on every container recreate, and Ring lists each new id as
    another authorized device."""
    path = Path(state_dir) / HARDWARE_ID_FILE
    try:
        existing = path.read_text().strip()
        if existing:
            return existing
    except OSError:
        pass
    hid = str(uuid.uuid4())
    _write_private(path, hid)
    return hid


def token_updater(state_dir: str):
    path = Path(state_dir) / TOKEN_FILE

    def _update(token: dict[str, Any]) -> None:
        _write_private(path, json.dumps(token))

    return _update


def matches_device(device: Any, wanted: str) -> bool:
    """RETINA_SOURCE names the doorbell by id or by its name in the Ring app."""
    w = (wanted or "").strip()
    if not w:
        return True
    return w == str(getattr(device, "device_api_id", "")) or w.casefold() == str(getattr(device, "name", "")).casefold()


def listener_alive(listener: Any) -> bool:
    """RingEventListener.started stays True after the FCM client terminates
    itself; its receiver knows the truth."""
    if listener is None or not getattr(listener, "started", False):
        return False
    receiver = getattr(listener, "_receiver", None)
    is_started = getattr(receiver, "is_started", None)
    return bool(is_started()) if callable(is_started) else True


class RingEventFrameSource:
    def __init__(
        self,
        device: str,
        *,
        state_dir: str,
        burst_seconds: float = 15.0,
        cooldown_seconds: float = 30.0,
        event_kinds: tuple[str, ...] = DEFAULT_EVENT_KINDS,
        self_look_max_per_day: int = 8,
        self_look_min_spacing_seconds: float = 300.0,
        event_sink: Callable[[RingPush], None] | None = None,
    ) -> None:
        if not state_dir:
            raise ValueError("ring source requires RETINA_RING_STATE_DIR")
        self.device_wanted = device
        self.state_dir = state_dir
        self.burst_seconds = float(burst_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.event_kinds = tuple(normalize_kind(k) for k in event_kinds if k.strip())
        self.self_look_max_per_day = int(self_look_max_per_day)
        self.self_look_min_spacing_seconds = float(self_look_min_spacing_seconds)
        # Called (sync, on the loop) once per recorded push. RetinaService
        # sets it to publish the doorbell vision/street events.
        self.event_sink = event_sink

        self._ring: Any = None
        self._device: Any = None
        self._listener: Any = None
        self._connected_at = 0.0
        self._burst_task: asyncio.Task | None = None
        self._supervise_task: asyncio.Task | None = None
        self._last_burst_end = 0.0
        self._frame: Any = None
        self._frame_ts = 0.0
        self._frame_served_ts = 0.0
        self._event_kind: str | None = None
        self._burst_correlation_id: str | None = None
        # Monotonic time a ring pushed the open live view out to; 0 = none.
        self._burst_extend_until = 0.0
        # True only while the frame loop runs and will read _burst_extend_until.
        self._frames_open = False
        # A ding that came while the live view was connecting or closing.
        self._pending_ding_corr: str | None = None
        self._closing_for_stop = False
        self._ding_reopen_times: list[float] = []
        self._self_look_day: str | None = None
        self._last_self_look_ts = 0.0

        # Read by RetinaService (capture_once / health).
        self.ever_connected = False
        self.last_error: str | None = None
        self.events_seen = 0
        self.events_ignored = 0
        self.bursts_started = 0
        self.bursts_failed = 0
        # Live views in a row that delivered no frame; 0 after any frame.
        self.consecutive_failed_bursts = 0
        self.last_event_kind: str | None = None
        self.last_event_ts: float | None = None
        # Every recorded push, whatever happened to the live view.
        self.dings_total = 0
        self.motions_total = 0
        self.last_ding_ts: float | None = None
        self.bursts_extended = 0
        self.ding_reopens_refused = 0
        self.self_looks_today = 0
        self.self_looks_total = 0
        self.last_self_look_ts: float | None = None
        self.last_self_look_reason: str | None = None

    def _has_new_frame(self) -> bool:
        return self._frame is not None and self._frame_ts > self._frame_served_ts

    @property
    def waiting_for_event(self) -> bool:
        """Listening and no unread frame: the normal state between events (and
        while a live view is still opening), not a failed read. False while
        unauthenticated or while the push listener is down."""
        return self.ever_connected and not self._has_new_frame()

    async def start(self) -> None:
        self._closing_for_stop = False
        if self._supervise_task is None or self._supervise_task.done():
            self._supervise_task = asyncio.create_task(self._supervise())

    async def stop(self) -> None:
        self._closing_for_stop = True
        tasks = [t for t in (self._supervise_task, self._burst_task) if t and not t.done()]
        for task in tasks:
            task.cancel()
        # Let the burst's finally close the live view before auth goes away.
        await asyncio.gather(*tasks, return_exceptions=True)
        self._supervise_task = self._burst_task = None
        await self._disconnect()

    async def _disconnect(self) -> None:
        self.ever_connected = False
        listener, ring = self._listener, self._ring
        self._listener = self._ring = self._device = None
        if listener is not None:
            try:
                await listener.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[RETINA] ring listener stop failed: {exc}")
        if ring is not None:
            try:
                await ring.auth.async_close()
            except Exception:  # noqa: BLE001
                pass

    async def _supervise(self) -> None:
        """Connect, then keep the push link honest for the life of the source."""
        while True:
            if not self.ever_connected:
                await self._connect_once()
            elif not listener_alive(self._listener):
                self.last_error = "ring push listener stopped; reconnecting"
                logger.warning(f"[RETINA] {self.last_error}")
                await self._reconnect_when_idle()
            elif time.time() - self._connected_at >= RECONNECT_EVERY_SECONDS:
                logger.info("[RETINA] ring periodic reconnect (session refresh)")
                await self._reconnect_when_idle()
            await asyncio.sleep(SUPERVISE_SECONDS)

    async def _reconnect_when_idle(self) -> None:
        if self._burst_task is not None and not self._burst_task.done():
            return  # never pull auth out from under a live view; next round
        await self._disconnect()
        await self._connect_once()

    async def _connect_once(self) -> bool:
        try:
            from ring_doorbell import Auth, Ring, RingEventListener
        except ImportError as exc:
            # Outside the try below this killed _supervise silently and health
            # read the generic "connecting to camera".
            self.last_error = f"ring library missing in this image: {exc}"
            logger.error(f"[RETINA] {self.last_error}")
            return False

        token = _read_json(Path(self.state_dir) / TOKEN_FILE)
        if not token:
            self.last_error = "ring not authenticated: run `python -m app.ring_login` in this container"
            logger.error(f"[RETINA] {self.last_error}")
            return False
        try:
            auth = Auth(USER_AGENT, token, token_updater(self.state_dir), hardware_id=hardware_id(self.state_dir))
            self._ring = ring = Ring(auth)
            await ring.async_create_session()
            await ring.async_update_data()
            devices = ring.devices()
            cams = list(devices.doorbots) + list(devices.authorized_doorbots) + list(devices.stickup_cams)
            found = [d for d in cams if matches_device(d, self.device_wanted)]
            if not found:
                names = ", ".join(f"{d.name} ({d.device_api_id})" for d in cams) or "none"
                raise RuntimeError(f"no Ring camera matches RETINA_SOURCE={self.device_wanted!r}; account has: {names}")
            self._device = found[0]

            fcm_path = Path(self.state_dir) / FCM_FILE
            listener = RingEventListener(
                ring,
                _read_json(fcm_path),
                lambda creds: _write_private(fcm_path, json.dumps(creds)),
            )
            self._listener = listener
            listener.add_notification_callback(self.on_event)
            if not await listener.start():
                raise RuntimeError("Ring push listener did not start (FCM check-in or subscription refused)")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._disconnect()
            self.last_error = f"ring connect failed: {exc}"
            logger.error(f"[RETINA] {self.last_error}")
            return False
        self.ever_connected = True
        self._connected_at = time.time()
        self.last_error = None
        logger.info(f"[RETINA] ring listening for {self.event_kinds} on {self._device.name} ({self._device.device_api_id})")
        return True

    def on_event(self, event: Any) -> None:
        """RingEventListener callback (sync, on the event loop).

        Every push of a configured kind is recorded (counted, logged with a
        correlation id, handed to event_sink), whatever the live view does.
        A ding while a live view is open extends it; a ding in the cooldown
        starts a new one (2026-10-20: walk up = motion burst, then the press
        was dropped as "burst running or in cooldown")."""
        if self._device is None or getattr(event, "doorbot_id", None) != self._device.device_api_id:
            return
        if getattr(event, "is_update", False):
            return  # the same event again (Ring re-sends on state change)
        raw_kind = str(getattr(event, "kind", "") or "")
        kind = normalize_kind(raw_kind)
        if kind not in self.event_kinds:
            # Logged so a new spelling of a ding shows up instead of vanishing.
            logger.info(f"[RETINA] ring event kind {raw_kind!r} not in {self.event_kinds}; ignored")
            return
        now = time.time()
        self.events_seen += 1
        self.last_event_kind = kind
        self.last_event_ts = now
        if kind == "ding":
            self.dings_total += 1
            self.last_ding_ts = now
        elif kind == "motion":
            self.motions_total += 1
        corr = str(uuid.uuid4())
        live_view = self._live_view_for(kind, now, corr)
        ring_id = getattr(event, "id", None)
        try:
            occurred_at = float(getattr(event, "now", None) or now)
        except (TypeError, ValueError):
            occurred_at = now
        logger.info(
            f"[RETINA] ring {kind} recorded corr={corr} ring_event_id={ring_id} "
            f"raw_kind={raw_kind!r} live_view={live_view}"
        )
        if self.event_sink is not None:
            push = RingPush(
                kind=kind,
                raw_kind=raw_kind,
                ring_event_id=str(ring_id) if ring_id is not None else None,
                occurred_at=occurred_at,
                received_ts=now,
                correlation_id=corr,
                live_view=live_view,
            )
            try:
                self.event_sink(push)
            except Exception as exc:  # noqa: BLE001 -- recording must not break the listener
                logger.warning(f"[RETINA] ring event sink failed corr={corr}: {type(exc).__name__}: {exc}")

    def _burst_running(self) -> bool:
        return self._burst_task is not None and not self._burst_task.done()

    def _live_view_for(self, kind: str, now: float, corr: str) -> str:
        if self._burst_running():
            if kind != "ding":
                self.events_ignored += 1
                return "already_open"
            if self._frames_open:
                self._extend_burst(corr)
                return "extended"
            # Connecting (frames start covers it) or closing (_burst's finally
            # reopens): never marked extended with no frame loop to read it.
            self._pending_ding_corr = corr
            return "queued"
        if self.should_start_burst(now):
            self._start_burst(kind, corr)
            return "started"
        if kind != "ding":
            self.events_ignored += 1
            return "cooldown"
        if not self._take_ding_reopen(now):
            self.events_ignored += 1
            return "ding_capped"
        self._start_burst(kind, corr)
        return "started"

    def _take_ding_reopen(self, now: float) -> bool:
        """Quota for ding-opened live views that skip the cooldown."""
        self._ding_reopen_times = [t for t in self._ding_reopen_times if now - t < DING_REOPEN_WINDOW_SECONDS]
        if len(self._ding_reopen_times) >= DING_REOPEN_MAX:
            self.ding_reopens_refused += 1
            return False
        self._ding_reopen_times.append(now)
        return True

    def _extend_burst(self, corr: str) -> None:
        self.bursts_extended += 1
        self._burst_extend_until = time.monotonic() + self.burst_seconds
        self._event_kind = "ding"
        self._burst_correlation_id = corr

    def _start_burst(self, kind: str, corr: str) -> None:
        logger.info(f"[RETINA] ring {kind}: opening live view for {self.burst_seconds:.0f}s corr={corr}")
        self._burst_extend_until = 0.0
        self._frames_open = False
        self._burst_correlation_id = corr
        self._burst_task = asyncio.create_task(self._burst(kind))

    def should_start_burst(self, now: float) -> bool:
        if self._burst_running():
            return False
        return self._last_burst_end == 0.0 or now - self._last_burst_end >= self.cooldown_seconds

    def self_look(self, reason: str, now: float | None = None) -> str:
        """Open a live view because Orion expects someone (spec section 5).

        Same live-view call as an event burst: ring_doorbell's
        generate_async_webrtc_stream takes only the device and an SDP offer,
        no ding id. Returns the outcome: started | not_ready | burst_running |
        cooldown | daily_cap | spacing."""
        now = time.time() if now is None else now
        day = datetime.fromtimestamp(now, tz=timezone.utc).date().isoformat()
        if day != self._self_look_day:
            self._self_look_day = day
            self.self_looks_today = 0
        if not self.ever_connected or self._device is None:
            return "not_ready"
        if self._burst_running():
            return "burst_running"
        if not self.should_start_burst(now):
            return "cooldown"
        if self.self_looks_today >= self.self_look_max_per_day:
            return "daily_cap"
        if self._last_self_look_ts and now - self._last_self_look_ts < self.self_look_min_spacing_seconds:
            return "spacing"
        self.self_looks_today += 1
        self.self_looks_total += 1
        self._last_self_look_ts = now
        self.last_self_look_ts = now
        self.last_self_look_reason = reason
        self._start_burst(SELF_LOOK_KIND, f"self_look:{reason}")
        return "started"

    async def _burst(self, kind: str) -> None:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
        from aiortc.sdp import candidate_from_sdp

        ensure_video_jitter_capacity()
        self.bursts_started += 1
        self._event_kind = kind
        started = time.time()
        device = self._device
        loop = asyncio.get_running_loop()
        answer: asyncio.Future = loop.create_future()
        remote_candidates: list[tuple[str, int]] = []
        remote_set = False
        pending: set[asyncio.Task] = set()
        video_track: asyncio.Future = loop.create_future()
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(urls=device.get_ice_servers())]))
        session_id = uuid.uuid4().hex

        def on_message(msg: Any) -> None:
            if msg.error_code is not None:
                if not answer.done():
                    answer.set_exception(RuntimeError(f"Ring refused the live view: {msg.error_code} {msg.error_message or ''}".strip()))
            elif msg.answer is not None:
                if not answer.done():
                    answer.set_result(msg.answer)
            elif msg.candidate:
                item = (msg.candidate, int(msg.sdp_m_line_index or 0))
                if remote_set:  # trickled after the answer
                    task = loop.create_task(add_candidate(*item))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                else:
                    remote_candidates.append(item)

        async def add_candidate(cand: str, mline: int) -> None:
            try:
                c = candidate_from_sdp(cand.split(":", 1)[1] if cand.startswith("candidate:") else cand)
                c.sdpMLineIndex = mline
                await pc.addIceCandidate(c)
            except Exception as exc:  # noqa: BLE001
                logger.info(f"[RETINA] ring ICE candidate skipped: {exc}")

        @pc.on("track")
        def _on_track(track):  # noqa: ANN001
            if track.kind == "video" and not video_track.done():
                video_track.set_result(track)

        phase = "offer"
        try:
            pc.addTransceiver("video", direction="recvonly")
            pc.addTransceiver("audio", direction="recvonly")  # Ring's live_view always asks for audio
            await pc.setLocalDescription(await pc.createOffer())
            # Registers the session BEFORE talking to Ring, so the finally's
            # close_webrtc_stream always finds it (the sync variant registers
            # only after an answer and leaks the websocket on a timeout).
            await device.generate_async_webrtc_stream(
                pc.localDescription.sdp,
                session_id,
                on_message,
                keep_alive_timeout=int(self.burst_seconds * MAX_BURST_FACTOR) + 30,
            )
            phase = "answer"
            sdp = await asyncio.wait_for(answer, timeout=ANSWER_TIMEOUT_SECONDS)
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))
            remote_set = True
            for item in remote_candidates:
                await add_candidate(*item)

            phase = "frames"
            frames_start = time.monotonic()
            deadline = frames_start + self.burst_seconds
            cap = frames_start + self.burst_seconds * MAX_BURST_FACTOR
            track = await asyncio.wait_for(video_track, timeout=self.burst_seconds)
            self._frames_open = True
            if self._pending_ding_corr is not None:
                # A ding while connecting: this full burst, from now, covers it.
                self._event_kind = "ding"
                self._burst_correlation_id = self._pending_ding_corr
                self._pending_ding_corr = None

            def until() -> float:
                # A ding while open moves _burst_extend_until (on_event).
                return min(cap, max(deadline, self._burst_extend_until))

            while (remaining := until() - time.monotonic()) > 0:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    if until() - time.monotonic() > 0:
                        continue  # extended while waiting for this frame
                    raise
                self._frame = frame
                self._frame_ts = time.time()
                self.consecutive_failed_bursts = 0
            self._frames_open = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- incl. TimeoutError and aiortc MediaStreamError
            if self._frame_ts < started:
                self.bursts_failed += 1
                self.consecutive_failed_bursts += 1
                if phase == "answer" and isinstance(exc, asyncio.TimeoutError):
                    why = f"no answer from Ring within {ANSWER_TIMEOUT_SECONDS:.0f}s"
                elif phase == "frames" and isinstance(exc, asyncio.TimeoutError):
                    why = f"no frame within {self.burst_seconds:.0f}s"
                else:
                    why = f"{type(exc).__name__}: {exc}"
                self.last_error = f"ring {kind} live view failed ({why})"
                logger.warning(f"[RETINA] {self.last_error}")
            # Frames already arrived: the timeout is the burst's normal end,
            # and a stream that ends early still delivered.
        finally:
            self._frames_open = False
            for task in list(pending):
                task.cancel()
            await pc.close()
            try:
                await device.close_webrtc_stream(session_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[RETINA] ring live view close failed: {exc}")
            self._last_burst_end = time.time()
            self._frame = None
            corr = self._pending_ding_corr
            self._pending_ding_corr = None
            if corr is not None and not self._closing_for_stop:
                # A ding came while this view connected-and-failed or closed.
                if self._take_ding_reopen(time.time()):
                    logger.info(f"[RETINA] ring ding queued during live view close: reopening corr={corr}")
                    self._start_burst("ding", corr)
                else:
                    self.events_ignored += 1
                    logger.info(f"[RETINA] ring ding queued corr={corr}: reopen refused (ding quota)")

    @property
    def blind(self) -> bool:
        """Ring events arrive but live views keep delivering no picture."""
        return self.consecutive_failed_bursts >= BLIND_AFTER_FAILED_BURSTS

    async def read(self) -> FrameReadResult | None:
        """Newest live-view frame not yet served, else None."""
        if not self._has_new_frame():
            return None
        frame, ts = self._frame, self._frame_ts
        self._frame_served_ts = ts
        bgr = await asyncio.to_thread(frame.to_ndarray, format="bgr24")
        h, w = bgr.shape[:2]
        if self.last_error and self.last_error.startswith("ring "):
            self.last_error = None
        return FrameReadResult(
            frame=bgr,
            ts=ts,
            width=w,
            height=h,
            source_meta={
                "type": "ring",
                "ring_event_kind": self._event_kind,
                "ring_correlation_id": self._burst_correlation_id,
            },
        )
