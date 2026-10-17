"""Event-driven frames from a Ring doorbell (RETINA_SOURCE_TYPE=ring).

A battery Ring doorbell with no Ring Protect plan has no RTSP, no saved
recordings, and cannot take a snapshot while it records a motion event
(https://github.com/dgreif/ring/wiki/Snapshot-Limitations). The one picture
source left is a live view. So: listen for Ring's push events (motion, ding);
on each one, open a short WebRTC live view, keep the newest decoded frame,
and hang up after RETINA_RING_BURST_SECONDS. RetinaService.capture_loop
samples that frame at RETINA_FPS exactly as it samples an RTSP camera, so
downstream sees the same frame-pointer stream as the walkway, only in bursts.

Never continuous: no live view runs without an event, one burst at a time,
and RETINA_RING_COOLDOWN_SECONDS between the end of a burst and the next
start (each live view wakes the camera and costs battery).

Auth: run ``python -m app.ring_login`` once inside the container. It writes
the refresh token under RETINA_RING_STATE_DIR; this source keeps it fresh.
Without a token the source reports ``ring not authenticated`` and checks
again every CONNECT_RETRY_SECONDS, so no restart is needed after login.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from .sources import FrameReadResult

USER_AGENT = "orion-vision-retina"
TOKEN_FILE = "ring_token.json"
FCM_FILE = "ring_fcm_credentials.json"
HARDWARE_ID_FILE = "ring_hardware_id"
# Ring push kinds that mean "someone or something is at the door".
DEFAULT_EVENT_KINDS = ("motion", "ding")
CONNECT_RETRY_SECONDS = 60.0


def _write_private(path: Path, data: str) -> None:
    """Atomic write, owner-only: these files are Ring account credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(data)
    os.replace(tmp, path)


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


class RingEventFrameSource:
    def __init__(
        self,
        device: str,
        *,
        state_dir: str,
        burst_seconds: float = 15.0,
        cooldown_seconds: float = 30.0,
        event_kinds: tuple[str, ...] = DEFAULT_EVENT_KINDS,
    ) -> None:
        if not state_dir:
            raise ValueError("ring source requires RETINA_RING_STATE_DIR")
        self.device_wanted = device
        self.state_dir = state_dir
        self.burst_seconds = float(burst_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.event_kinds = tuple(k.strip().lower() for k in event_kinds if k.strip())

        self._ring: Any = None
        self._device: Any = None
        self._listener: Any = None
        self._burst_task: asyncio.Task | None = None
        self._connect_task: asyncio.Task | None = None
        self._last_burst_end = 0.0
        self._frame: Any = None
        self._frame_ts = 0.0
        self._frame_served_ts = 0.0
        self._event_kind: str | None = None

        # Read by RetinaService (capture_once / health).
        self.ever_connected = False
        self.last_error: str | None = None
        self.events_seen = 0
        self.events_ignored = 0
        self.bursts_started = 0
        self.bursts_failed = 0
        self.last_event_kind: str | None = None
        self.last_event_ts: float | None = None

    def _has_new_frame(self) -> bool:
        return self._frame is not None and self._frame_ts > self._frame_served_ts

    @property
    def waiting_for_event(self) -> bool:
        """Connected and no unread frame: the normal state between events (and
        while a live view is still opening), not a failed read."""
        return self.ever_connected and not self._has_new_frame()

    async def start(self) -> None:
        if self._connect_task is None or self._connect_task.done():
            self._connect_task = asyncio.create_task(self._connect())

    async def stop(self) -> None:
        for task in (self._burst_task, self._connect_task):
            if task and not task.done():
                task.cancel()
        if self._listener is not None:
            try:
                await self._listener.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[RETINA] ring listener stop failed: {exc}")
        self._listener = None
        if self._ring is not None:
            try:
                await self._ring.auth.async_close()
            except Exception:  # noqa: BLE001
                pass
        self._ring = None
        self.ever_connected = False

    async def _connect(self) -> None:
        while not await self._connect_once():
            await asyncio.sleep(CONNECT_RETRY_SECONDS)

    async def _connect_once(self) -> bool:
        from ring_doorbell import Auth, Ring, RingEventListener

        token = _read_json(Path(self.state_dir) / TOKEN_FILE)
        if not token:
            self.last_error = "ring not authenticated: run `python -m app.ring_login` in this container"
            logger.error(f"[RETINA] {self.last_error}")
            return False
        try:
            auth = Auth(USER_AGENT, token, token_updater(self.state_dir), hardware_id=hardware_id(self.state_dir))
            ring = Ring(auth)
            await ring.async_create_session()
            await ring.async_update_data()
            devices = ring.devices()
            cams = list(devices.doorbots) + list(devices.authorized_doorbots) + list(devices.stickup_cams)
            found = [d for d in cams if matches_device(d, self.device_wanted)]
            if not found:
                names = ", ".join(f"{d.name} ({d.device_api_id})" for d in cams) or "none"
                raise RuntimeError(f"no Ring camera matches RETINA_SOURCE={self.device_wanted!r}; account has: {names}")
            self._ring, self._device = ring, found[0]

            fcm_path = Path(self.state_dir) / FCM_FILE
            listener = RingEventListener(
                ring,
                _read_json(fcm_path),
                lambda creds: _write_private(fcm_path, json.dumps(creds)),
            )
            listener.add_notification_callback(self.on_event)
            if not await listener.start():
                raise RuntimeError("Ring push listener did not start (FCM check-in or subscription refused)")
            self._listener = listener
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"ring connect failed: {exc}"
            logger.error(f"[RETINA] {self.last_error}")
            if self._ring is not None:
                try:
                    await self._ring.auth.async_close()
                except Exception:  # noqa: BLE001
                    pass
            self._ring = self._device = None
            return False
        self.ever_connected = True
        self.last_error = None
        logger.info(f"[RETINA] ring listening for {self.event_kinds} on {self._device.name} ({self._device.device_api_id})")
        return True

    def on_event(self, event: Any) -> None:
        """RingEventListener callback (sync, on the event loop)."""
        if self._device is None or getattr(event, "doorbot_id", None) != self._device.device_api_id:
            return
        if getattr(event, "is_update", False):
            return  # the same event again (Ring re-sends on state change)
        kind = str(getattr(event, "kind", "") or "").lower()
        if kind not in self.event_kinds:
            return
        self.events_seen += 1
        self.last_event_kind = kind
        self.last_event_ts = time.time()
        if not self.should_start_burst(time.time()):
            self.events_ignored += 1
            logger.info(f"[RETINA] ring {kind} ignored: burst running or in cooldown")
            return
        logger.info(f"[RETINA] ring {kind}: opening live view for {self.burst_seconds:.0f}s")
        self._burst_task = asyncio.create_task(self._burst(kind))

    def should_start_burst(self, now: float) -> bool:
        if self._burst_task is not None and not self._burst_task.done():
            return False
        return self._last_burst_end == 0.0 or now - self._last_burst_end >= self.cooldown_seconds

    async def _burst(self, kind: str) -> None:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

        self.bursts_started += 1
        self._event_kind = kind
        started = time.time()
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(urls=self._device.get_ice_servers())]))
        session_id: str | None = None
        try:
            pc.addTransceiver("video", direction="recvonly")
            pc.addTransceiver("audio", direction="recvonly")  # Ring's live_view always offers audio
            video_track: asyncio.Future = asyncio.get_running_loop().create_future()

            @pc.on("track")
            def _on_track(track):  # noqa: ANN001
                if track.kind == "video" and not video_track.done():
                    video_track.set_result(track)

            await pc.setLocalDescription(await pc.createOffer())
            offer = pc.localDescription.sdp
            from ring_doorbell.webrtcstream import RingWebRtcStream

            session_id = RingWebRtcStream.get_sdp_session_id(offer)
            answer = await self._device.generate_webrtc_stream(offer)
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))

            deadline = time.monotonic() + self.burst_seconds
            track = await asyncio.wait_for(video_track, timeout=self.burst_seconds)
            while (remaining := deadline - time.monotonic()) > 0:
                frame = await asyncio.wait_for(track.recv(), timeout=remaining)
                self._frame = frame
                self._frame_ts = time.time()
        except asyncio.TimeoutError:
            if self._frame_ts < started:
                self.bursts_failed += 1
                self.last_error = f"ring {kind} live view delivered no frame in {self.burst_seconds:.0f}s"
                logger.warning(f"[RETINA] {self.last_error}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.bursts_failed += 1
            self.last_error = f"ring {kind} live view failed: {exc}"
            logger.warning(f"[RETINA] {self.last_error}")
        finally:
            await pc.close()
            if session_id and self._device is not None:
                try:
                    await self._device.close_webrtc_stream(session_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[RETINA] ring live view close failed: {exc}")
            self._last_burst_end = time.time()
            self._frame = None

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
            source_meta={"type": "ring", "ring_event_kind": self._event_kind},
        )
