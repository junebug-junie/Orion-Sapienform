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


class _BlockingCap(_FakeCap):
    """read() blocks until released by the test: a dead RTSP socket."""

    def __init__(self, gate: threading.Event) -> None:
        super().__init__()
        self.gate = gate

    def read(self):
        self.entered = getattr(self, "entered", 0) + 1
        self.gate.wait()
        return super().read()


def test_a_stop_that_times_out_never_leaves_two_live_readers():
    # Review finding: a shared stop flag cleared by the next start() revived a
    # reader still blocked in read(), leaving two RTSP sessions decoding.
    gate = threading.Event()
    caps: list[_FakeCap] = []

    def opener(_url):
        cap = _BlockingCap(gate) if not caps else _FakeCap()
        caps.append(cap)
        return cap

    src = RtspFrameSource("rtsp://x", opener=opener, join_timeout=0.05)

    async def run():
        await src.start()
        assert _wait(lambda: caps and getattr(caps[0], "entered", 0) >= 1)  # first reader is blocked
        await src.stop()                                     # times out
        await src.start()                                    # second reader
        assert _wait(lambda: len(caps) >= 2 and caps[1].i >= 3)
        gate.set()                                           # dead socket finally returns
        assert _wait(lambda: caps[0].released)               # old reader exits, not revived
        alive = [t for t in threading.enumerate() if t.name == "retina-rtsp-reader"]
        await src.stop()
        return alive

    alive = asyncio.run(run())
    assert len(alive) == 1


def test_a_decoder_exception_reconnects_instead_of_killing_the_reader():
    calls = {"n": 0}

    class _Boom(_FakeCap):
        def read(self):
            raise RuntimeError("corrupt stream")

    def opener(_url):
        calls["n"] += 1
        return _Boom() if calls["n"] == 1 else _FakeCap()

    src = RtspFrameSource("rtsp://x", opener=opener, reconnect_seconds=0.01)

    async def run():
        await src.start()
        assert _wait(lambda: calls["n"] >= 2)
        assert _wait(lambda: src._latest is not None)
        got = await src.read()
        await src.stop()
        return got

    got = asyncio.run(run())
    assert got is not None
    assert src.last_error and "RuntimeError" in src.last_error
