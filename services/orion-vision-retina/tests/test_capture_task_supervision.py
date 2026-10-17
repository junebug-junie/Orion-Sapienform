"""2026-10-17 walkway outage: the capture task died silently and health said ok
for 3 h.

Chain: cleanup_old_frames raced another pruner on the shared frames dir
(isfile -> file deleted -> getmtime raised FileNotFoundError), the exception
escaped capture_loop (cleanup ran outside its try), the task ended, and the
service held a reference to it so asyncio never logged "exception was never
retrieved". Health kept publishing status ok with frames_published frozen.

These tests pin each link: the pruner survives the race and only touches its
own files, capture_loop survives a cleanup failure, a dead forever-task exits
the process loudly, and health/watchdog stop lying about a dead publisher.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_REPO = str(ROOT.parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from app import frame_store  # noqa: E402
from app.frame_store import cleanup_old_frames, owned_frame_pattern  # noqa: E402
from app.health import RetinaMetrics, make_system_health_envelope  # noqa: E402
from app.settings import Settings  # noqa: E402


# ---------------------------------------------------------------- pruner ----


def _touch(d: Path, name: str, age: float) -> Path:
    p = d / name
    p.write_bytes(b"x")
    t = time.time() - age
    os.utime(p, (t, t))
    return p


class _RacingScandir:
    """Real os.scandir, but another pruner deletes each file right after it
    is listed and before the caller looks at it: the live race."""

    real = os.scandir

    def __init__(self, path):
        self._it = _RacingScandir.real(path)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._it.close()

    def __iter__(self):
        for e in self._it:
            os.remove(e.path)
            yield e


def test_cleanup_survives_files_vanishing_between_listing_and_stat(tmp_path, monkeypatch):
    for i in range(3):
        _touch(tmp_path, f"walkway_walkway_17922615752{i}0_0123abc{i}.jpg", 600)
    monkeypatch.setattr(frame_store.os, "scandir", _RacingScandir)
    assert cleanup_old_frames(str(tmp_path), 300, camera_id="walkway", stream_id="walkway") == 0


def test_cleanup_prunes_only_this_camera_and_stream(tmp_path):
    mine_old = _touch(tmp_path, "walkway_walkway_1792261575200_0123abcd.jpg", 600)
    mine_new = _touch(tmp_path, "walkway_walkway_1792261575999_0123abce.jpg", 10)
    other_stream = _touch(tmp_path, "walkway_side_1792261575200_0123abcd.jpg", 600)
    doorbell = _touch(tmp_path, "doorbell_front_1792261575200_0123abcd.jpg", 600)
    edge = _touch(tmp_path, "frame_1792261575263_44e535.jpg", 600)
    still = _touch(tmp_path, "still_1792261575263_44e535.jpg", 600)

    assert cleanup_old_frames(str(tmp_path), 300, camera_id="walkway", stream_id="walkway") == 1

    assert not mine_old.exists() and mine_new.exists()
    assert other_stream.exists() and doorbell.exists() and edge.exists() and still.exists()


def test_owned_pattern_matches_save_frame_names(tmp_path):
    import numpy as np

    saved = frame_store.save_frame(
        np.zeros((4, 4, 3), dtype=np.uint8),
        directory=str(tmp_path),
        camera_id="walk way",
        stream_id="walkway",
        ts=time.time(),
        quality=80,
    )
    assert owned_frame_pattern("walk way", "walkway").match(os.path.basename(saved.image_path))
    assert not owned_frame_pattern("walkway", "walkway").match(os.path.basename(saved.image_path))


def test_cleanup_missing_dir_is_a_no_op(tmp_path):
    assert cleanup_old_frames(str(tmp_path / "nope"), 1, camera_id="a", stream_id="b") == 0


# --------------------------------------------------------------- service ----


class _Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, env):
        self.published.append((channel, env))


class _Src:
    """Minimal source; attributes are set per test."""

    stopped = 0

    async def read(self):
        return None

    async def start(self):
        pass

    async def stop(self):
        self.stopped += 1


def _svc(**overrides):
    from app.main import RetinaService

    settings = Settings(_env_file=None, RETINA_SOURCE_TYPE="folder", **overrides)
    svc = RetinaService(settings=settings, bus=_Bus())
    svc.source = _Src()
    svc.exit_codes = []
    svc._exit = svc.exit_codes.append
    return svc


def test_capture_loop_keeps_running_when_cleanup_raises(monkeypatch):
    """The live failure: cleanup raising must not end the capture task."""
    import app.main as main_mod

    def boom(*_a, **_k):
        raise FileNotFoundError("frame vanished")

    monkeypatch.setattr(main_mod, "cleanup_old_frames", boom)
    svc = _svc(RETINA_FPS=50.0)
    calls = []

    async def capture_once():
        calls.append(1)
        return False

    svc.capture_once = capture_once

    async def run():
        task = svc._spawn_forever(svc.capture_loop(), "capture_loop")
        await asyncio.sleep(0.2)
        assert not task.done(), task.exception() if task.done() else None
        svc._shutdown.set()
        await asyncio.wait_for(task, 2)

    asyncio.run(run())
    assert len(calls) >= 3
    assert svc.exit_codes == []


@pytest.mark.parametrize("how", ["raises", "returns", "cancelled"])
def test_a_forever_task_ending_exits_the_process(how):
    svc = _svc()

    async def dies():
        await asyncio.sleep(0)
        if how == "raises":
            raise FileNotFoundError("frame vanished")

    async def run():
        task = svc._spawn_forever(dies() if how != "cancelled" else asyncio.sleep(10), "capture_loop")
        if how == "cancelled":
            await asyncio.sleep(0)
            task.cancel()
        try:
            await task
        except (FileNotFoundError, asyncio.CancelledError):
            pass
        for _ in range(3):
            await asyncio.sleep(0)  # done callback, then the exit task

    asyncio.run(run())
    assert svc.exit_codes == [1]
    assert svc.source.stopped == 1  # camera released (RTSP TEARDOWN) before exit


def test_two_tasks_dying_exit_once():
    svc = _svc()

    async def boom():
        raise RuntimeError("x")

    async def run():
        a = svc._spawn_forever(boom(), "capture_loop")
        b = svc._spawn_forever(boom(), "health_loop")
        await asyncio.gather(a, b, return_exceptions=True)
        for _ in range(3):
            await asyncio.sleep(0)

    asyncio.run(run())
    assert svc.exit_codes == [1]


def test_wedged_rtsp_readers_exit_through_the_same_path():
    svc = _svc()
    svc.source.needs_process_restart = True
    svc.source.stall_error = "rtsp stalled 200s; 2 abandoned readers"
    assert asyncio.run(svc.capture_once()) is False
    assert svc.exit_codes == [1] and svc.source.stopped == 1


def test_stop_does_not_exit_the_process():
    svc = _svc()

    async def run():
        svc._capture_task = svc._spawn_forever(asyncio.sleep(10), "capture_loop")
        svc._health_task = svc._spawn_forever(asyncio.sleep(10), "health_loop")
        await asyncio.sleep(0)
        svc._shutdown.set()
        for t in (svc._capture_task, svc._health_task):
            t.cancel()
        await asyncio.gather(svc._capture_task, svc._health_task, return_exceptions=True)
        await asyncio.sleep(0)

    asyncio.run(run())
    assert svc.exit_codes == []


# ------------------------------------------------------- health/watchdog ----


def test_publish_stall_after_the_limit():
    svc = _svc()
    t0 = svc._started_at
    assert svc.publish_stall(t0 + 100) is None
    reason = svc.publish_stall(t0 + 130)
    assert reason and "no frame published for 130s" in reason
    svc._last_publish_ts = t0 + 120
    assert svc.publish_stall(t0 + 130) is None


def test_publish_stall_ignores_rtsp_quiet_wait_and_restarts_the_clock():
    svc = _svc()
    t0 = svc._started_at
    svc.source.quiet_wait_remaining = lambda: 42.0
    assert svc.publish_stall(t0 + 500) is None
    svc.source.quiet_wait_remaining = lambda: 0.0
    # The clock restarted when the wait was last seen, not at boot.
    assert svc.publish_stall(t0 + 600) is None
    assert svc.publish_stall(t0 + 621) is not None


def test_publish_stall_ignores_an_event_camera_between_events():
    svc = _svc()
    svc.source.waiting_for_event = True
    assert svc.publish_stall(svc._started_at + 10_000) is None


def test_publish_stall_disabled_by_zero():
    svc = _svc(RETINA_PUBLISH_STALL_SECONDS=0)
    assert svc.publish_stall(svc._started_at + 10_000) is None
    assert svc.publisher_dead(svc._started_at + 10_000) is None


def test_watchdog_exits_when_rtsp_streams_and_the_loop_publishes_nothing():
    """Loop ticking, no capture error, reader streaming, nothing published
    (a lost hand-off between reader and loop). The 2026-10-17 dead loop
    itself is caught earlier, by the done callback."""
    svc = _svc()
    now = svc._started_at + 10_255
    mono = 1_000_000.0
    svc._loop_tick_mono = mono
    svc._last_publish_ts = svc._started_at + 1
    svc.source.last_frame_age = lambda: 0.4
    dead = svc.publisher_dead(now, mono)
    assert dead and "source is streaming" in dead


def test_watchdog_does_not_exit_for_failures_a_restart_cannot_fix():
    """Disk full / bus down / upload refused while the camera streams:
    degraded, not a restart loop that churns camera sessions."""
    svc = _svc()
    now = svc._started_at + 500
    mono = 1_000_000.0
    svc._loop_tick_mono = mono
    svc._last_capture_error_mono = mono - 3
    svc.source.last_frame_age = lambda: 0.4
    assert svc.publish_stall(now) is not None
    assert svc.publisher_dead(now, mono) is None


def test_capture_loop_records_a_capture_error():
    svc = _svc(RETINA_FPS=50.0)

    async def capture_once():
        raise OSError("No space left on device")

    svc.capture_once = capture_once

    async def run():
        task = asyncio.ensure_future(svc.capture_loop())
        await asyncio.sleep(0.05)
        svc._shutdown.set()
        await asyncio.wait_for(task, 2)

    asyncio.run(run())
    assert svc._last_capture_error_mono is not None
    assert "No space left" in svc.metrics.last_error


def test_watchdog_quiet_during_a_long_heal_backoff():
    """Heal backoff can wait up to 8 x 90 s: quiet wait, not a stall."""
    svc = _svc()
    mono = 1_000_000.0
    svc._loop_tick_mono = mono
    svc.source.quiet_wait_remaining = lambda: 600.0
    svc.source.last_frame_age = lambda: 700.0
    now = svc._started_at + 720
    assert svc.publish_stall(now) is None
    assert svc.publisher_dead(now, mono) is None


def test_watchdog_does_not_exit_when_the_camera_is_down():
    svc = _svc()
    now = svc._started_at + 1_000
    svc._loop_tick_mono = mono = 1_000_000.0
    svc.source.last_frame_age = lambda: 900.0  # reader gets nothing either
    assert svc.publish_stall(now) is not None  # degraded, honestly
    assert svc.publisher_dead(now, mono) is None  # but a restart would not help


def test_watchdog_exits_when_capture_loop_stopped_ticking():
    svc = _svc()
    svc._loop_tick_mono = 1_000_000.0
    assert svc.publisher_dead(svc._started_at + 1, 1_000_000.0 + 100) is None
    dead = svc.publisher_dead(svc._started_at + 1, 1_000_000.0 + 121)
    assert dead and "capture_loop has not ticked" in dead


def test_wall_clock_jump_alone_is_not_a_hang():
    """NTP step / laptop resume: wall time jumps, monotonic does not."""
    svc = _svc()
    svc._loop_tick_mono = time.monotonic()
    svc._last_publish_ts = time.time()
    assert svc.publisher_dead(time.time() + 7_200, time.monotonic()) is None


def test_health_loop_reports_degraded_then_exits_once():
    svc = _svc(HEALTH_INTERVAL_SECONDS=10.0)
    old = time.time() - 10_000
    svc._started_at = svc._stall_clock_ts = svc._last_publish_ts = old
    svc._loop_tick_mono = time.monotonic()
    svc.source.last_frame_age = lambda: 0.5
    svc.metrics.frames_published = 89625
    svc.metrics.last_frame_ts = old

    async def run():
        task = svc._spawn_forever(svc._health_loop(), "health_loop")
        await asyncio.wait_for(task, 2)
        await asyncio.sleep(0)

    asyncio.run(run())
    assert svc.exit_codes == [1]  # watchdog exit; the task ending does not exit twice
    assert svc.source.stopped == 1
    (_, env), = svc.bus.published
    payload = env.payload
    assert payload["status"] == "degraded"
    assert payload["details"]["capture_stalled"] is True
    assert "no frame published" in payload["details"]["capture_stalled_reason"]


def test_health_envelope_status_ok_without_a_stall():
    env = make_system_health_envelope(
        service_name="s",
        service_version="v",
        camera_id="c",
        stream_id="t",
        source_type="rtsp",
        source_ok=True,
        metrics=RetinaMetrics(),
        fps_target=1.0,
        storage_dir="/x",
        heartbeat_interval_sec=10.0,
    )
    assert env.payload["status"] == "ok"
    assert env.payload["details"]["capture_stalled"] is False


def test_settings_reject_a_stall_limit_inside_normal_pauses():
    with pytest.raises(ValueError, match="RETINA_PUBLISH_STALL_SECONDS"):
        Settings(_env_file=None, RETINA_PUBLISH_STALL_SECONDS=20)
    with pytest.raises(ValueError, match="RETINA_PUBLISH_STALL_SECONDS"):
        Settings(_env_file=None, RETINA_PUBLISH_STALL_SECONDS=-1)
    with pytest.raises(ValueError, match="RETINA_PUBLISH_STALL_SECONDS"):
        # stop + 30 s clip + start can hold the loop past 31 s
        Settings(_env_file=None, RETINA_CLIP_ENABLED=True, RETINA_PUBLISH_STALL_SECONDS=31)
    assert Settings(_env_file=None).RETINA_PUBLISH_STALL_SECONDS == 120.0
    assert Settings(_env_file=None, RETINA_PUBLISH_STALL_SECONDS=0).RETINA_PUBLISH_STALL_SECONDS == 0
