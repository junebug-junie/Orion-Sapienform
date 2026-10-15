"""A live RTSP source hands out the newest frame, never a backlog.

Calling cap.read() once per tick on a 30 fps stream returns the oldest
buffered frame, so the picture drifts behind. RtspFrameSource drains the
stream on a thread and serves only the latest frame, once.
"""

import asyncio
import threading
import time

import numpy as np

from app.sources import RtspFrameSource


class _FakeCap:
    """Emits frames 0, 1, 2, ... as fast as it is read; frame value = index."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.i = 0
        self.fail_after = fail_after
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:
        return not self.released

    def set(self, *_a) -> bool:
        return True

    def read(self):
        self.reads += 1
        time.sleep(0.001)
        if self.fail_after is not None and self.i >= self.fail_after:
            return False, None
        frame = np.full((4, 6, 3), self.i % 256, dtype=np.uint8)
        self.i += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def _wait(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_read_returns_the_newest_frame_not_the_oldest():
    cap = _FakeCap()
    src = RtspFrameSource("rtsp://x", opener=lambda _url: cap)

    async def run():
        await src.start()
        assert _wait(lambda: cap.i >= 50)
        got = await src.read()
        await src.stop()
        return got

    got = asyncio.run(run())
    assert got is not None and (got.width, got.height) == (6, 4)
    # Many frames were decoded; the one handed out is near the head, not frame 0.
    assert int(got.frame[0, 0, 0]) >= 40


def test_a_frame_is_served_once_then_none_until_a_new_one():
    cap = _FakeCap(fail_after=5)  # five frames, then the stream stalls
    src = RtspFrameSource("rtsp://x", opener=lambda _url: cap, reconnect_seconds=60)

    async def run():
        await src.start()
        assert _wait(lambda: cap.i >= 5)
        first = await src.read()
        second = await src.read()
        await src.stop()
        return first, second

    first, second = asyncio.run(run())
    assert first is not None and int(first.frame[0, 0, 0]) == 4
    assert second is None  # a stalled camera never re-serves an old picture


def test_stop_releases_the_capture():
    cap = _FakeCap()
    src = RtspFrameSource("rtsp://x", opener=lambda _url: cap)

    async def run():
        await src.start()
        assert _wait(lambda: cap.i >= 3)
        await src.stop()

    asyncio.run(run())
    assert cap.released
    assert not any(t.name == "retina-rtsp-reader" for t in threading.enumerate())
