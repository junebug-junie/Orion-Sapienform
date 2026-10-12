import cv2, threading, time
import logging
from typing import Callable, Optional
from orion.vision.stream_ids import strip_userinfo

from .settings import get_settings

settings = get_settings()
logger = logging.getLogger("orion-vision-edge.capture_source")

# 2026-10-11: cam0 dropped its RTSP connection at 02:00 (the camera's scheduled reboot). This
# reader never reopened the stream, failed 200k reads in a row, and get_frame() kept returning
# the last good picture -- so for 16 h every consumer saw a dark, empty room as if it were live,
# and Juniper's visit to the basement was never seen. Now a frame older than STALE_AFTER_SEC is
# not handed out at all, and the stream is reopened after REOPEN_AFTER_SEC of failed reads
# (backing off to REOPEN_BACKOFF_MAX_SEC while the camera stays down).
STALE_AFTER_SEC = 5.0
REOPEN_AFTER_SEC = 5.0
REOPEN_BACKOFF_MAX_SEC = 60.0
# FFmpeg's own defaults let a half-dead RTSP connection block an open or a read for ~30 s.
STREAM_TIMEOUT_MS = 5000


def _open_capture(source: str):
    if "://" in source:
        return cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS],
        )
    return cv2.VideoCapture(source)


class CameraSource:
    def __init__(
        self,
        source: str,
        *,
        opener: Optional[Callable[[str], object]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.source = source
        # Log-safe form: the RTSP URL carries the camera password.
        self._source_for_log = strip_userinfo(source) if "://" in source else source
        self.cap = None
        self.last_frame = None
        self.last_frame_at: Optional[float] = None
        self.reopens = 0
        self._opener = opener or _open_capture
        self._clock = clock
        self._failing_since: Optional[float] = None
        self._next_reopen = 0.0
        self._backoff = REOPEN_AFTER_SEC
        self._retry = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.width = settings.WIDTH
        self.height = settings.HEIGHT
        self.fps = settings.FPS

    def _open(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing camera before reopen: {e}")
        if self._stop.is_set():
            self.cap = None
            return
        self.cap = self._opener(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera source: {self._source_for_log}")
            return
        logger.info(f"Camera source opened: {self._source_for_log}")
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def start(self):
        logger.info(f"Opening camera source: {self._source_for_log}")
        self._open()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def step(self) -> float:
        """One read attempt (reopening the stream when it has been failing too long); returns
        how long to wait before the next one."""
        now = self._clock()
        try:
            ok, frame = self.cap.read() if self.cap is not None and self.cap.isOpened() else (False, None)
        except Exception as e:  # noqa: BLE001 -- a decoder error is a failed read, not a dead thread
            logger.warning(f"Camera read raised {type(e).__name__}; counting it as a failed read.")
            ok, frame = False, None
        if ok and frame is not None:
            # Frame before timestamp, and frame_age_sec reads the timestamp first: a reader can
            # pair a newer frame with an older age, never a stale frame with a fresh age.
            self.last_frame = frame
            self.last_frame_at = now
            if self._failing_since is not None:
                logger.info(f"Camera stream recovered after {now - self._failing_since:.1f}s.")
            self._failing_since, self._backoff, self._retry = None, REOPEN_AFTER_SEC, 0
            return 0.5 / max(self.fps, 1)
        self._retry += 1
        if self._failing_since is None:
            self._failing_since = now
            self._next_reopen = now + REOPEN_AFTER_SEC
        if self._retry % 20 == 1:
            logger.warning(f"Failed to read frame from camera (attempt {self._retry}).")
        if now >= self._next_reopen:
            self.reopens += 1
            logger.warning(
                f"Reopening camera after {now - self._failing_since:.0f}s without a frame (reopen #{self.reopens})."
            )
            try:
                self._open()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Camera reopen raised {type(e).__name__}.")
            # From after the open: a slow open must not use up the wait before the next one.
            self._next_reopen = self._clock() + self._backoff
            self._backoff = min(self._backoff * 2, REOPEN_BACKOFF_MAX_SEC)
        return 0.25

    def _run(self):
        while not self._stop.is_set():
            try:
                wait = self.step()
            except Exception as e:  # noqa: BLE001 -- backstop: a dead reader thread blinds the camera for good
                logger.warning(f"Camera reader step raised {type(e).__name__}.")
                wait = 0.25
            self._stop.wait(wait)
        # The reader owns the capture: releasing it here, not from stop(), avoids freeing it
        # while a read is still blocked inside it.
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("Camera released.")
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")

    def frame_age_sec(self) -> Optional[float]:
        if self.last_frame_at is None:
            return None
        return self._clock() - self.last_frame_at

    def get_frame(self):
        """The latest frame, or None once it is older than STALE_AFTER_SEC: a stale frame handed
        out as live is worse than no frame (every caller already handles None)."""
        age = self.frame_age_sec()
        if age is None or age > STALE_AFTER_SEC:
            return None
        return self.last_frame

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
