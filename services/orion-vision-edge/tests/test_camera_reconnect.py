"""The camera reader reconnects and never hands out a stale frame (2026-10-11: cam0 rebooted at
02:00, the reader never reopened, and a frozen dark frame was served as live for 16 h)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture import REOPEN_AFTER_SEC, STALE_AFTER_SEC, CameraSource


class _Cap:
    def __init__(self, reads):
        self.reads = list(reads)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return self.reads.pop(0) if self.reads else (False, None)

    def set(self, *_a):
        return True

    def release(self):
        self.released = True


FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def _camera(caps, t):
    opened = []

    def opener(_src):
        cap = caps.pop(0)
        opened.append(cap)
        return cap

    cam = CameraSource("rtsp://u:p@cam/Preview_01_sub", opener=opener, clock=lambda: t["now"])
    cam._open()
    return cam, opened


def test_a_dead_stream_stops_serving_its_last_frame_and_reopens() -> None:
    t = {"now": 100.0}
    dead = _Cap([(True, FRAME)])          # one good frame, then the camera reboots
    alive = _Cap([(True, FRAME)])
    cam, opened = _camera([dead, alive], t)

    cam.step()
    assert cam.get_frame() is FRAME

    t["now"] += 1.0
    cam.step()                             # first failure
    assert cam.get_frame() is FRAME        # 1 s old: still live

    t["now"] += STALE_AFTER_SEC            # failing for 6 s
    assert cam.get_frame() is None         # never served as live past the limit
    cam.step()
    assert cam.reopens == 1 and dead.released and opened[-1] is alive

    t["now"] += 0.1
    cam.step()
    assert cam.get_frame() is FRAME and cam.frame_age_sec() == 0.0


def test_reopen_backs_off_while_the_camera_stays_down() -> None:
    t = {"now": 0.0}
    caps = [_Cap([]) for _ in range(10)]
    cam, _ = _camera(caps, t)
    reopen_times = []
    for _ in range(400):                   # 100 s of failed reads at 0.25 s
        before = cam.reopens
        cam.step()
        if cam.reopens != before:
            reopen_times.append(t["now"])
        t["now"] += 0.25
    gaps = [b - a for a, b in zip(reopen_times, reopen_times[1:])]
    assert reopen_times[0] == REOPEN_AFTER_SEC
    assert gaps[:3] == [5.0, 10.0, 20.0] and max(gaps) <= 60.0


def test_no_frame_since_boot_reads_as_none() -> None:
    t = {"now": 0.0}
    cam, _ = _camera([_Cap([])], t)
    assert cam.get_frame() is None and cam.frame_age_sec() is None
