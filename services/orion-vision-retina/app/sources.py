from __future__ import annotations

import asyncio
import collections
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from .frame_corruption import frame_report


class FrameReadResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame: Any
    ts: float
    width: int
    height: int
    source_meta: dict[str, Any] = Field(default_factory=dict)


class FrameSource(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def read(self) -> FrameReadResult | None: ...


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class FolderFrameSource:
    def __init__(self, directory: str) -> None:
        if not directory:
            raise ValueError("folder source requires RETINA_SOURCE directory")
        self.directory = directory
        self._files: list[str] = []
        self._index = 0

    def _refresh_files(self) -> None:
        if not os.path.isdir(self.directory):
            self._files = []
            return
        self._files = sorted(
            os.path.join(self.directory, name)
            for name in os.listdir(self.directory)
            if Path(name).suffix.lower() in _IMAGE_EXTS
        )

    async def start(self) -> None:
        self._refresh_files()
        self._index = 0

    async def stop(self) -> None:
        return None

    async def read(self) -> FrameReadResult | None:
        if not self._files:
            self._refresh_files()
        if not self._files:
            return None
        path = self._files[self._index % len(self._files)]
        self._index += 1
        frame = await asyncio.to_thread(cv2.imread, path)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return FrameReadResult(
            frame=frame,
            ts=time.time(),
            width=w,
            height=h,
            source_meta={"path": path, "type": "folder"},
        )


class MockFrameSource(FolderFrameSource):
    """Test/dev: random sample from folder like legacy mock."""

    async def read(self) -> FrameReadResult | None:
        if not self._files:
            return None
        path = random.choice(self._files)
        frame = await asyncio.to_thread(cv2.imread, path)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return FrameReadResult(
            frame=frame,
            ts=time.time(),
            width=w,
            height=h,
            source_meta={"path": path, "type": "mock"},
        )


class _VideoCaptureSource:
    def __init__(
        self,
        source: str | int,
        *,
        width: int | None,
        height: int | None,
        reconnect_seconds: float,
        source_type: str,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._reconnect_seconds = reconnect_seconds
        self._source_type = source_type
        self._cap: cv2.VideoCapture | None = None
        self._failures = 0

    async def start(self) -> None:
        await self._open()

    async def stop(self) -> None:
        if self._cap is not None:
            await asyncio.to_thread(self._cap.release)
            self._cap = None

    async def _open(self) -> None:
        await self.stop()
        cap = await asyncio.to_thread(cv2.VideoCapture, self._source)
        if self._width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        self._cap = cap

    async def read(self) -> FrameReadResult | None:
        if self._cap is None or not self._cap.isOpened():
            await self._open()
        if self._cap is None or not self._cap.isOpened():
            self._failures += 1
            return None
        ok, frame = await asyncio.to_thread(self._cap.read)
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= 3:
                await asyncio.sleep(self._reconnect_seconds)
                await self._open()
                self._failures = 0
            return None
        self._failures = 0
        h, w = frame.shape[:2]
        return FrameReadResult(
            frame=frame,
            ts=time.time(),
            width=w,
            height=h,
            source_meta={"type": self._source_type},
        )


_RTSP_TRANSPORTS = {"tcp", "udp", "http"}


def rtsp_capture_options(transport: str, io_timeout_seconds: float) -> str:
    """The OPENCV_FFMPEG_CAPTURE_OPTIONS string for an RTSP camera.

    ``timeout`` is the RTSP demuxer's socket I/O timeout in microseconds. It is
    the one name the FFmpeg bundled with this image's OpenCV (4.10, avformat
    59.27 = FFmpeg 5.1) honours: measured in the built image against a fake
    RTSP server that goes quiet mid-stream, ``timeout`` cut the blocked read to
    5.1s while ``stimeout`` and ``rw_timeout`` were silently ignored (30s, the
    OpenCV default). TCP interleaving stops the UDP packet loss/reordering that
    smeared the bottom of the walkway frames (``RTP: bad cseq``)."""
    t = (transport or "tcp").strip().lower()
    if t not in _RTSP_TRANSPORTS:
        raise ValueError(f"unsupported RTSP transport: {transport!r}")
    micros = max(1, int(float(io_timeout_seconds) * 1_000_000))
    return f"rtsp_transport;{t}|timeout;{micros}"


def _open_rtsp_capture(url: str, *, open_timeout_ms: int, read_timeout_ms: int):
    return cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_ms),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout_ms),
        ],
    )


class RtspSessionGate:
    """Quiet period between RTSP sessions to one camera.

    Root cause, proven live 2026-10-16 on the walkway Reolink: when a client
    drops its session and reconnects quickly, the camera keeps the old
    session alive and muxes it into the new one. The stream turns to corrupt
    RTP (thousands of ``RTP: PT=60/61: bad cseq`` lines a minute, two
    interleaved sequence series) and every decoded frame smears, for hours.
    Stopping the retina, waiting 180 s and starting it again gave a clean
    stream with no camera reboot. So no (re)connect may start until
    ``quiet_seconds`` have passed since this process last held, or might have
    held, a session with the camera.

    "Activity" is any moment a session may exist: an open attempt, every
    decoded frame, a release, a reader abandoned while still blocked. The
    newest activity time is persisted (atomic write, throttled while
    streaming, immediate on release) so a container restart -- including a
    deploy, which is how the 2026-10-16 incident started -- also waits. If
    the file is missing or unreadable at boot, the time is unknown and the
    gate waits one full quiet period.
    """

    PERSIST_EVERY_SECONDS = 5.0

    def __init__(self, quiet_seconds: float, state_path: str | None = None, *, clock=time.time) -> None:
        self.quiet_seconds = max(0.0, float(quiet_seconds))
        self.state_path = state_path or None
        self._clock = clock
        self._lock = threading.Lock()
        self._not_before = 0.0
        self._last_persist: float | None = None
        self._persist_failed = False
        # Wall time a session last ended (a release); None until one does.
        self.last_session_end_ts: float | None = None
        self._last_activity, self.boot_state = self._load()

    def _load(self) -> tuple[float | None, str]:
        now = self._clock()
        if self.quiet_seconds <= 0:
            return None, "disabled"
        if not self.state_path:
            return now, "unknown"
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                ts = float(json.load(fh)["last_session_activity_ts"])
        except FileNotFoundError:
            return now, "unknown"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning(f"[RETINA] rtsp session state unreadable ({exc}); waiting a full quiet period")
            return now, "unknown"
        # A clock that went backwards (or a future value) must not shorten the wait.
        return min(ts, now), "persisted"

    def touch(self, *, ended: bool = False) -> None:
        """A session may exist right now. ``ended``: one was just released."""
        if self.quiet_seconds <= 0:
            return
        now = self._clock()
        with self._lock:
            self._last_activity = now
            if ended:
                self.last_session_end_ts = now
            due = ended or self._last_persist is None or now - self._last_persist >= self.PERSIST_EVERY_SECONDS
            if due:
                self._last_persist = now
        if due:
            self._persist(now)

    def defer(self, seconds: float) -> None:
        """No connect for at least ``seconds`` from now (longer heal waits)."""
        with self._lock:
            self._not_before = max(self._not_before, self._clock() + max(0.0, float(seconds)))

    def remaining(self) -> float:
        """Seconds until a new connect may start (0 = now)."""
        if self.quiet_seconds <= 0:
            return 0.0
        now = self._clock()
        with self._lock:
            last = self._last_activity
            not_before = self._not_before
        rem = 0.0 if last is None else self.quiet_seconds - (now - last)
        return max(0.0, rem, not_before - now)

    def _persist(self, ts: float) -> None:
        if not self.state_path:
            return
        tmp = f"{self.state_path}.tmp"
        try:
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"last_session_activity_ts": ts}, fh)
            os.replace(tmp, self.state_path)
            self._persist_failed = False
        except OSError as exc:
            if not self._persist_failed:
                logger.warning(
                    f"[RETINA] cannot persist rtsp session state to {self.state_path} ({exc}); "
                    "a restart will wait a full quiet period"
                )
            self._persist_failed = True


class CorruptionTracker:
    """Decides when a stream has been corrupt long enough to drop the session.

    Records one verdict per checked frame. ``record`` returns True when the
    checks span at least ``window_seconds`` since the last reset, at least
    ``min_samples`` fall inside the window, and the corrupt share inside it is
    at least ``min_share``. On 2026-10-16 every walkway frame smeared for 10
    hours, so a high share over minutes is the incident shape; a few bad
    frames (a bird, rain on the lens) never reach it.
    """

    def __init__(self, window_seconds: float, min_share: float, *, min_samples: int = 5, clock=time.monotonic) -> None:
        self.window_seconds = float(window_seconds)
        self.min_share = float(min_share)
        self.min_samples = int(min_samples)
        self._clock = clock
        self._samples: collections.deque[tuple[float, bool]] = collections.deque()
        self._since: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._since = None

    def _prune(self, now: float) -> None:
        while self._samples and now - self._samples[0][0] > self.window_seconds:
            self._samples.popleft()

    def corrupt_rate(self) -> float | None:
        self._prune(self._clock())
        if not self._samples:
            return None
        return sum(c for _, c in self._samples) / len(self._samples)

    def window_full(self) -> bool:
        return self._since is not None and self._clock() - self._since >= self.window_seconds

    def record(self, corrupt: bool) -> bool:
        now = self._clock()
        if self._since is None:
            self._since = now
        self._samples.append((now, bool(corrupt)))
        self._prune(now)
        if not self.window_full() or len(self._samples) < self.min_samples:
            return False
        return (self.corrupt_rate() or 0.0) >= self.min_share


class RtspFrameSource(_VideoCaptureSource):
    """A live network camera, sampled at RETINA_FPS.

    A live RTSP stream arrives at the camera's own frame rate (the walkway
    Reolink's main stream is 30 fps). Calling ``cap.read()`` once per tick at
    a lower RETINA_FPS returns the OLDEST buffered frame, not the current one,
    so the picture falls further behind the longer the service runs. A
    background thread therefore keeps reading the stream and holds only the
    newest frame; ``read()`` hands that one out, once.

    Stall watchdog (incident 2026-10-15: the walkway reader went silent for
    ~34 min, no log, until a container restart). ``read()`` checks how long it
    has been since the reader thread last delivered a frame. Past
    ``stall_seconds`` it logs a WARNING, sets that thread's own stop event,
    abandons it (never joins it -- it may be wedged inside FFmpeg), and starts
    a fresh reader on a fresh capture. A stopped thread can never write
    ``_latest`` again: the stop event is set, and checked, under ``_lock``.
    """

    # Abandoned readers may still hold a camera session until their blocked
    # read returns. Reolinks limit sessions; do not stack more than this.
    MAX_ABANDONED_ALIVE = 2
    WEDGED_WINDOWS_BEFORE_EXIT = 4

    def __init__(
        self,
        url: str,
        *,
        width: int | None = None,
        height: int | None = None,
        reconnect_seconds: float = 5.0,
        opener=None,
        join_timeout: float | None = None,
        stall_seconds: float = 30.0,
        transport: str = "tcp",
        io_timeout_seconds: float = 5.0,
        reconnect_quiet_seconds: float = 0.0,
        session_state_path: str | None = None,
        gate: RtspSessionGate | None = None,
        heal_enabled: bool = False,
        heal_check_every: int = 10,
        heal_window_seconds: float = 120.0,
        heal_min_share: float = 0.8,
        heal_max_backoff: int = 8,
        frame_scorer=None,
    ) -> None:
        if not url:
            raise ValueError("rtsp source requires RETINA_SOURCE url")
        super().__init__(
            url,
            width=width,
            height=height,
            reconnect_seconds=reconnect_seconds,
            source_type="rtsp",
        )
        # Process-wide, read by OpenCV's FFmpeg backend at open time. Set here,
        # on the constructing thread, never concurrently with an open. Only
        # the RTSP source sets it; webcam/folder/mock never touch it.
        self.capture_options = rtsp_capture_options(transport, io_timeout_seconds)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.capture_options
        read_ms = max(1, int(float(io_timeout_seconds) * 1000))
        open_ms = max(read_ms * 2, 10_000)
        self._opener = opener or (
            lambda u: _open_rtsp_capture(u, open_timeout_ms=open_ms, read_timeout_ms=read_ms)
        )
        self._lock = threading.Lock()
        self._latest: tuple[object, float] | None = None
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._abandoned: list[threading.Thread] = []
        # stop() must outlast a reader blocked in an open (open_ms) or a read
        # (read_ms) so the reader itself releases the capture -- that release
        # is what sends RTSP TEARDOWN. A thread still blocked when the process
        # exits dies with no TEARDOWN and the camera keeps the session.
        self._join_timeout = (
            float(join_timeout) if join_timeout is not None else max(open_ms, read_ms) / 1000.0 + 2.0
        )
        self._gate = gate or RtspSessionGate(reconnect_quiet_seconds, session_state_path)
        # True while the live reader waits out the quiet period; the stall
        # watchdog must not count that wait as a stall.
        self._in_quiet_wait = False
        self.heal_enabled = bool(heal_enabled)
        self._heal_check_every = max(1, int(heal_check_every))
        self._heal_max_backoff = max(1, int(heal_max_backoff))
        self._tracker = CorruptionTracker(heal_window_seconds, heal_min_share)
        self._score = frame_scorer or frame_report
        self._served = 0
        self.corruption_checks = 0
        self.corrupt_frames = 0
        self.corruption_heals = 0
        self._consecutive_heals = 0
        self.stall_seconds = float(stall_seconds)
        # True once any frame has ever arrived; lets the service tell
        # "still connecting" apart from "stream stalled".
        self.ever_connected = False
        self.last_error: str | None = None
        # Monotonic time of the newest frame any live reader delivered, and of
        # the last reader (re)start. The stall clock runs from the later one,
        # so a fresh reader gets a full stall window to connect.
        self._last_frame_mono: float | None = None
        self._reader_started_mono: float = time.monotonic()
        self.stall_restarts = 0
        self._stalled_since: float | None = None
        # Monotonic time the live reader last finished a step. Frames missing
        # while this keeps moving = camera down/garbage: the reader reconnects
        # on its own. Frames missing AND this frozen = reader wedged in FFmpeg.
        self._reader_progress_mono: float = time.monotonic()
        # Consecutive watchdog windows refused by MAX_ABANDONED_ALIVE. At
        # WEDGED_WINDOWS_BEFORE_EXIT the process asks to be restarted.
        self._capped_windows = 0
        self.needs_process_restart = False
        # Set while a stall is in progress, cleared when frames resume.
        # last_error stays sticky (the most recent error ever seen).
        self.stall_error: str | None = None

    def _open_sync(self):
        cap = self._opener(self._source)
        if self._width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        return cap

    def _reader(self, stop_evt: threading.Event) -> None:
        """Owns one capture. Each thread has its OWN stop event, so a thread
        left blocked in cap.read() on a dead socket after a timed-out stop()
        can never be revived by the next start() clearing a shared flag."""
        cap = None
        failures = 0
        try:
            while not stop_evt.is_set():
                try:
                    if cap is None or not cap.isOpened():
                        if cap is not None:
                            self._release(cap)
                            cap = None
                        if not self._quiet_wait(stop_evt):
                            break
                        # The open itself may create a camera-side session.
                        self._gate.touch()
                        cap = self._open_sync()
                        self._progress(stop_evt)
                        if not cap.isOpened():
                            if not stop_evt.is_set():
                                self.last_error = "rtsp open failed"
                            stop_evt.wait(self._reconnect_seconds)
                            self._progress(stop_evt)
                            continue
                    ok, frame = cap.read()
                    self._progress(stop_evt)
                except Exception as exc:  # noqa: BLE001 -- a decoder error must not kill the eye
                    self._progress(stop_evt)
                    if not stop_evt.is_set():
                        self.last_error = f"rtsp reader error: {type(exc).__name__}: {exc}"
                        logger.warning(f"[RETINA] {self.last_error}; reconnecting")
                    if cap is not None:
                        self._release(cap)
                    cap = None
                    stop_evt.wait(self._reconnect_seconds)
                    continue
                if not ok or frame is None:
                    failures += 1
                    if failures >= 3:
                        if not stop_evt.is_set():
                            self.last_error = "rtsp read failed 3 times; reconnecting"
                            logger.warning(f"[RETINA] {self.last_error}")
                        self._release(cap)
                        cap = None
                        failures = 0
                        stop_evt.wait(self._reconnect_seconds)
                        self._progress(stop_evt)
                    continue
                failures = 0
                self._gate.touch()  # a live session (persisted, throttled)
                with self._lock:
                    # Checked under the lock the watchdog sets it under: once
                    # a reader is stopped, it can never publish a frame.
                    if stop_evt.is_set():
                        break
                    self._latest = (frame, time.time())
                    self._last_frame_mono = time.monotonic()
                    self.ever_connected = True
        finally:
            if cap is not None:
                self._release(cap)

    def _release(self, cap) -> None:
        """Close the capture (FFmpeg sends RTSP TEARDOWN; verified in the
        image against a fake server) and start the quiet-period clock."""
        try:
            cap.release()
        except Exception as exc:  # noqa: BLE001 -- the clock must still start
            logger.warning(f"[RETINA] rtsp capture release failed: {type(exc).__name__}: {exc}")
        finally:
            self._gate.touch(ended=True)

    def _quiet_wait(self, stop_evt: threading.Event) -> bool:
        """Block (in 1 s steps, so stop stays fast) until the quiet period
        since the last session has passed. False if stopped meanwhile."""
        logged = False
        try:
            while not stop_evt.is_set():
                rem = self._gate.remaining()
                if rem <= 0:
                    break
                if not stop_evt.is_set():
                    self._in_quiet_wait = True
                if not logged:
                    logger.info(
                        f"[RETINA] rtsp quiet period: waiting {rem:.0f}s before connecting "
                        f"(quiet={self._gate.quiet_seconds:.0f}s since the last camera session; "
                        "a faster reconnect makes the camera mux the stale session in: bad cseq + smear)"
                    )
                    logged = True
                self._progress(stop_evt)
                stop_evt.wait(min(rem, 1.0))
        finally:
            if logged and not stop_evt.is_set():
                self._in_quiet_wait = False
                # The stall window starts at the real connect attempt, not at
                # the start of the wait.
                self._reader_started_mono = time.monotonic()
                self._progress(stop_evt)
        if logged and not stop_evt.is_set():
            logger.info("[RETINA] rtsp quiet period over; connecting")
        return not stop_evt.is_set()

    def quiet_wait_remaining(self) -> float:
        """Seconds the live reader still has to wait before connecting (0 if
        it is not waiting)."""
        return round(self._gate.remaining(), 1) if self._in_quiet_wait else 0.0

    @property
    def last_session_end_ts(self) -> float | None:
        return self._gate.last_session_end_ts

    def corrupt_frame_rate(self) -> float | None:
        return self._tracker.corrupt_rate()

    def _progress(self, stop_evt: threading.Event) -> None:
        """The live reader finished one step (an open, a read, a wait). An
        abandoned reader's steps do not count."""
        if not stop_evt.is_set():
            self._reader_progress_mono = time.monotonic()

    def _spawn_reader(self) -> None:
        self._stop_evt = threading.Event()
        self._in_quiet_wait = False
        self._reader_started_mono = time.monotonic()
        self._reader_progress_mono = self._reader_started_mono
        self._thread = threading.Thread(
            target=self._reader, args=(self._stop_evt,), name="retina-rtsp-reader", daemon=True
        )
        self._thread.start()

    async def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._spawn_reader()

    async def stop(self) -> None:
        with self._lock:
            self._stop_evt.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            await asyncio.to_thread(thread.join, self._join_timeout)
            if thread.is_alive():
                # Blocked in a read on a dead socket. Its own stop event is
                # set, so it exits (and releases its capture) when the read
                # returns; it is never handed a cleared flag again.
                self._abandoned.append(thread)
                self.last_error = "rtsp reader still blocked after stop; will exit on its next read"
                # It still holds a session; it restarts the clock again when
                # it finally releases.
                self._gate.touch()
        self._in_quiet_wait = False
        self._abandoned = [t for t in self._abandoned if t.is_alive()]

    def last_frame_age(self) -> float | None:
        """Seconds since a reader last delivered a frame (None before the first)."""
        last = self._last_frame_mono
        return None if last is None else max(0.0, time.monotonic() - last)

    def _check_stall(self) -> None:
        if not self.ever_connected or self.stall_seconds <= 0 or self._in_quiet_wait:
            return
        now = time.monotonic()
        last = self._last_frame_mono or 0.0
        if now - max(last, self._reader_started_mono) <= self.stall_seconds:
            return
        if self._stalled_since is None:
            self._stalled_since = last
        quiet = now - last
        if now - max(self._reader_progress_mono, self._reader_started_mono) <= self.stall_seconds:
            # The reader is still opening/reading/waiting and reconnects on
            # its own (camera down, garbage stream). Say so once; no restart.
            if self.stall_error is None:
                self.stall_error = f"rtsp stalled: no frame for {quiet:.0f}s; reader alive and reconnecting"
                self.last_error = self.stall_error
                logger.warning(f"[RETINA] {self.stall_error}")
            return
        # No frame AND the reader has not finished a step for a whole window:
        # it is wedged inside FFmpeg. Abandon it, open a fresh capture.
        self._abandoned = [t for t in self._abandoned if t.is_alive()]
        if len(self._abandoned) >= self.MAX_ABANDONED_ALIVE:
            self._capped_windows += 1
            self.last_error = self.stall_error = (
                f"rtsp stalled {quiet:.0f}s; {len(self._abandoned)} abandoned readers "
                "still blocked, not opening another camera session yet"
            )
            logger.error(f"[RETINA] {self.last_error}")
            if self._capped_windows >= self.WEDGED_WINDOWS_BEFORE_EXIT:
                self.needs_process_restart = True
            self._reader_started_mono = now  # re-check after another window
            return
        self._capped_windows = 0
        with self._lock:
            self._stop_evt.set()
        old, self._thread = self._thread, None
        if old is not None and old.is_alive():
            self._abandoned.append(old)
            # Wedged with a session open: the replacement waits the quiet
            # period from now (and again from its release, if that comes).
            self._gate.touch()
        self.stall_restarts += 1
        self.last_error = f"rtsp stalled: no frame for {quiet:.0f}s; reader wedged, restarted it"
        self.stall_error = self.last_error
        logger.warning(
            f"[RETINA] {self.last_error} (stall_restarts={self.stall_restarts}, "
            f"abandoned_alive={len(self._abandoned)})"
        )
        self._spawn_reader()

    async def read(self) -> FrameReadResult | None:
        self._check_stall()
        if self._thread is None or not self._thread.is_alive():
            await self.start()
        with self._lock:
            latest, self._latest = self._latest, None
        if latest is None:
            return None  # nothing new since the last tick
        if self._stalled_since is not None:
            logger.info(
                f"[RETINA] rtsp frames resumed after {time.monotonic() - self._stalled_since:.0f}s "
                f"(stall_restarts={self.stall_restarts})"
            )
            self._stalled_since = None
        self.stall_error = None
        self._capped_windows = 0
        frame, ts = latest
        if self.heal_enabled:
            self._served += 1
            if self._served % self._heal_check_every == 0:
                await self._check_corruption(frame)
        h, w = frame.shape[:2]
        return FrameReadResult(frame=frame, ts=ts, width=w, height=h, source_meta={"type": "rtsp"})

    async def _check_corruption(self, frame) -> None:
        """Score one frame with the eval's own math (app/frame_corruption.py)
        off the event loop; heal when corruption has persisted."""
        try:
            report = await asyncio.to_thread(self._score, frame)
        except Exception as exc:  # noqa: BLE001 -- a scoring bug must not stop the eye
            logger.warning(f"[RETINA] corruption check failed: {type(exc).__name__}: {exc}")
            return
        corrupt = bool(report.get("corrupt"))
        self.corruption_checks += 1
        self.corrupt_frames += int(corrupt)
        if self._tracker.record(corrupt):
            self._heal()
        elif self._tracker.window_full() and (self._tracker.corrupt_rate() or 0.0) < self._tracker.min_share:
            # A full healthy window: the last heal (if any) worked.
            self._consecutive_heals = 0

    def _heal(self) -> None:
        """Corrupt stream (stale camera session muxed in): drop the session,
        wait the quiet period, reconnect. Never reboots the camera. Each heal
        in a row that did not clear it doubles the wait (capped)."""
        rate = self._tracker.corrupt_rate() or 0.0
        self._consecutive_heals += 1
        self.corruption_heals += 1
        backoff = min(2 ** (self._consecutive_heals - 1), self._heal_max_backoff)
        wait = self._gate.quiet_seconds * backoff
        self.last_error = (
            f"rtsp stream corrupt: {rate:.0%} of checked frames over {self._tracker.window_seconds:.0f}s; "
            f"dropping the camera session and reconnecting after {wait:.0f}s quiet "
            f"(heal {self.corruption_heals}, {self._consecutive_heals} in a row)"
        )
        logger.warning(f"[RETINA] {self.last_error}")
        with self._lock:
            self._stop_evt.set()
            self._latest = None
        old, self._thread = self._thread, None
        if old is not None and old.is_alive():
            # Not wedged: frames are flowing, so it sees the stop on its next
            # read and releases (TEARDOWN). Tracked so a wedge would count.
            self._abandoned.append(old)
        self._gate.touch()
        self._gate.defer(wait)
        self._tracker.reset()
        self._spawn_reader()


class WebcamFrameSource(_VideoCaptureSource):
    def __init__(
        self,
        device: str,
        *,
        width: int | None = None,
        height: int | None = None,
        reconnect_seconds: float = 5.0,
    ) -> None:
        index: str | int = int(device) if device.isdigit() else device
        super().__init__(
            index,
            width=width,
            height=height,
            reconnect_seconds=reconnect_seconds,
            source_type="webcam",
        )


def create_frame_source(
    source_type: str,
    source: str,
    *,
    width: int | None = None,
    height: int | None = None,
    reconnect_seconds: float = 5.0,
    rtsp_transport: str = "tcp",
    rtsp_stall_seconds: float = 30.0,
    rtsp_io_timeout_seconds: float = 5.0,
    rtsp_reconnect_quiet_seconds: float = 0.0,
    rtsp_session_state_path: str | None = None,
    rtsp_heal_enabled: bool = False,
    rtsp_heal_check_every: int = 10,
    rtsp_heal_window_seconds: float = 120.0,
    rtsp_heal_min_share: float = 0.8,
) -> FrameSource:
    st = source_type.lower().strip()
    common = dict(width=width, height=height, reconnect_seconds=reconnect_seconds)
    if st == "folder":
        return FolderFrameSource(source)
    if st == "mock":
        return MockFrameSource(source)
    if st == "rtsp":
        return RtspFrameSource(
            source,
            **common,
            transport=rtsp_transport,
            stall_seconds=rtsp_stall_seconds,
            io_timeout_seconds=rtsp_io_timeout_seconds,
            reconnect_quiet_seconds=rtsp_reconnect_quiet_seconds,
            session_state_path=rtsp_session_state_path,
            heal_enabled=rtsp_heal_enabled,
            heal_check_every=rtsp_heal_check_every,
            heal_window_seconds=rtsp_heal_window_seconds,
            heal_min_share=rtsp_heal_min_share,
        )
    if st == "webcam":
        return WebcamFrameSource(source or "0", **common)
    raise ValueError(f"unsupported RETINA_SOURCE_TYPE: {source_type!r}")
