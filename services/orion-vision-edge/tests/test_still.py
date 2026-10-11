"""On-demand full-resolution still (app/still.py): the file is complete, and a dead stream is None."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.still import grab_still


class _Cap:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        f = self.frames.pop(0)
        return (f is not None), f

    def release(self):
        self.released = True


def test_writes_a_full_jpeg_after_early_failed_reads(tmp_path) -> None:
    frame = np.zeros((1920, 2560, 3), dtype=np.uint8)
    cap = _Cap([None, None, frame])
    ticks = iter([100.0, 102.5])
    out = grab_still("rtsp://x", str(tmp_path), opener=lambda _u, _t: cap, clock=lambda: next(ticks))
    assert out["width"] == 2560 and out["height"] == 1920
    assert out["grab_ms"] == 2500
    p = Path(out["image_path"])
    assert p.parent == tmp_path and p.read_bytes()[:2] == b"\xff\xd8"
    assert not list(tmp_path.glob(".*.tmp"))
    assert cap.released


def test_unopened_or_empty_stream_returns_none_and_releases(tmp_path) -> None:
    for cap in (_Cap([], opened=False), _Cap([None] * 10)):
        assert grab_still("rtsp://x", str(tmp_path), opener=lambda _u, _t, c=cap: c) is None
        assert cap.released
    assert not list(tmp_path.iterdir())


def test_gives_up_at_the_deadline(tmp_path, monkeypatch) -> None:
    import app.still as still

    t = {"now": 0.0}
    monkeypatch.setattr(still.time, "monotonic", lambda: t["now"])

    class _Slow(_Cap):
        def read(self):
            t["now"] += 5.0       # each read eats 5 s of an 8 s deadline
            return False, None

    cap = _Slow([])
    assert grab_still("rtsp://x", str(tmp_path), opener=lambda _u, _t: cap) is None
    assert t["now"] == 10.0 and cap.released   # two reads, then stop: no third
