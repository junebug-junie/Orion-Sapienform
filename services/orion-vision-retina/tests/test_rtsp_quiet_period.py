"""Quiet period between RTSP sessions, clean teardown, corruption self-heal.

Root cause (live 2026-10-16): the walkway Reolink keeps a dropped session
alive and muxes it into a fast reconnect (bad cseq, every frame smeared).
Every path that opens a camera session must wait the quiet period first, and
a corrupt stream must heal by dropping the session -- never a camera reboot.
"""

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.frame_corruption import frame_report
from app.settings import Settings
from app.sources import CorruptionTracker, RtspFrameSource, RtspSessionGate, create_frame_source

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class _Cap:
    """Frames 0, 1, 2 ... ; `fail_after` frames then reads fail."""

    def __init__(self, *, fail_after=None, value=None):
        self.i = 0
        self.fail_after = fail_after
        self.value = value
        self.released = False
        self.released_at = None

    def isOpened(self):
        return not self.released

    def set(self, *_a):
        return True

    def read(self):
        time.sleep(0.002)
        if self.fail_after is not None and self.i >= self.fail_after:
            return False, None
        v = self.i % 256 if self.value is None else self.value
        self.i += 1
        return True, np.full((4, 6, 3), v, dtype=np.uint8)

    def release(self):
        self.released = True
        self.released_at = time.monotonic()


class _Recorder:
    """Opener that logs open times and hands out caps from a factory."""

    def __init__(self, factory):
        self.factory = factory
        self.caps = []
        self.opened_at = []

    def __call__(self, _url):
        self.opened_at.append(time.monotonic())
        cap = self.factory(len(self.caps))
        self.caps.append(cap)
        return cap


def _wait(pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


# --- RtspSessionGate (fake clock) -------------------------------------------


def test_gate_waits_the_quiet_period_after_a_session(tmp_path):
    clock = Clock()
    gate = RtspSessionGate(90, str(tmp_path / "s.json"), clock=clock)
    clock.t += 1000  # unknown-at-boot wait long over
    assert gate.remaining() == 0
    gate.touch(ended=True)
    assert gate.remaining() == pytest.approx(90)
    clock.t += 60
    assert gate.remaining() == pytest.approx(30)
    clock.t += 31
    assert gate.remaining() == 0
    assert gate.last_session_end_ts == pytest.approx(clock.t - 91)


def test_gate_unknown_at_boot_waits_one_full_period(tmp_path):
    clock = Clock()
    gate = RtspSessionGate(90, str(tmp_path / "missing.json"), clock=clock)
    assert gate.boot_state == "unknown"
    assert gate.remaining() == pytest.approx(90)


def test_gate_unreadable_state_file_waits_one_full_period(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    gate = RtspSessionGate(90, str(p), clock=Clock())
    assert gate.boot_state == "unknown" and gate.remaining() == pytest.approx(90)


def test_gate_persisted_session_makes_a_restart_wait_the_remainder(tmp_path):
    p = tmp_path / "s.json"
    clock = Clock()
    first = RtspSessionGate(90, str(p), clock=clock)
    first.touch(ended=True)  # process 1 released its session ...
    assert json.loads(p.read_text())["last_session_activity_ts"] == clock.t
    clock.t += 30  # ... and the container restarted 30 s later
    second = RtspSessionGate(90, str(p), clock=clock)
    assert second.boot_state == "persisted"
    assert second.remaining() == pytest.approx(60)
    clock.t += 300  # restarted long after: no wait
    assert RtspSessionGate(90, str(p), clock=clock).remaining() == 0


def test_gate_future_timestamp_never_shortens_the_wait(tmp_path):
    p = tmp_path / "s.json"
    clock = Clock()
    p.write_text(json.dumps({"last_session_activity_ts": clock.t + 10_000}))
    assert RtspSessionGate(90, str(p), clock=clock).remaining() == pytest.approx(90)


def test_gate_persists_streaming_heartbeat_throttled_and_release_immediately(tmp_path):
    p = tmp_path / "s.json"
    clock = Clock()
    gate = RtspSessionGate(90, str(p), clock=clock)
    gate.touch()
    t0 = clock.t
    clock.t += 1
    gate.touch()  # throttled: a crash now loses at most PERSIST_EVERY_SECONDS
    assert json.loads(p.read_text())["last_session_activity_ts"] == t0
    clock.t += RtspSessionGate.PERSIST_EVERY_SECONDS
    gate.touch()
    assert json.loads(p.read_text())["last_session_activity_ts"] == clock.t
    clock.t += 0.5
    gate.touch(ended=True)  # release is always written
    assert json.loads(p.read_text())["last_session_activity_ts"] == clock.t


def test_gate_unwritable_state_dir_still_gates_in_memory(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    gate = RtspSessionGate(90, str(blocker / "sub" / "s.json"), clock=Clock())
    gate.touch(ended=True)  # warns, does not raise
    assert gate.remaining() == pytest.approx(90)


def test_gate_disabled_and_defer():
    clock = Clock()
    off = RtspSessionGate(0, None, clock=clock)
    off.touch(ended=True)
    assert off.remaining() == 0 and off.boot_state == "disabled"
    gate = RtspSessionGate(10, None, clock=clock)
    clock.t += 100
    gate.defer(40)
    assert gate.remaining() == pytest.approx(40)


# --- Every connect path waits the quiet period (real threads) ---------------

Q = 0.3  # quiet seconds for thread tests


def _gate_ready(quiet=Q):
    """A gate with no prior session (boot wait already over)."""
    clock = {"t": time.time() - 10_000}
    gate = RtspSessionGate(quiet, None, clock=lambda: clock["t"])
    # switch to the real clock after construction
    gate._clock = time.time
    return gate


def test_reader_reconnect_after_failed_reads_waits_the_quiet_period():
    rec = _Recorder(lambda n: _Cap(fail_after=2) if n == 0 else _Cap())
    src = RtspFrameSource("rtsp://x", opener=rec, reconnect_seconds=0.01, gate=_gate_ready())

    async def run():
        await src.start()
        assert _wait(lambda: len(rec.caps) >= 2)
        await src.stop()

    asyncio.run(run())
    released = rec.caps[0].released_at
    assert released is not None
    assert rec.opened_at[1] - released >= Q * 0.95


def test_process_start_waits_out_a_persisted_recent_session(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"last_session_activity_ts": time.time()}))
    rec = _Recorder(lambda n: _Cap())
    t0 = time.monotonic()
    src = RtspFrameSource("rtsp://x", opener=rec, reconnect_quiet_seconds=Q, session_state_path=str(p))

    async def run():
        await src.start()
        assert _wait(lambda: src.quiet_wait_remaining() > 0, timeout=0.2)
        assert _wait(lambda: rec.caps)
        await src.stop()

    asyncio.run(run())
    assert rec.opened_at[0] - t0 >= Q * 0.9
    # The release on stop is persisted for the next process.
    assert json.loads(p.read_text())["last_session_activity_ts"] >= time.time() - 5


def test_process_start_with_an_old_persisted_session_connects_immediately(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"last_session_activity_ts": time.time() - 3600}))
    rec = _Recorder(lambda n: _Cap())
    t0 = time.monotonic()
    src = RtspFrameSource("rtsp://x", opener=rec, reconnect_quiet_seconds=60, session_state_path=str(p))

    async def run():
        await src.start()
        assert _wait(lambda: rec.caps, timeout=1.0)
        await src.stop()

    asyncio.run(run())
    assert rec.opened_at[0] - t0 < 0.5


def test_stop_start_pause_path_waits_the_quiet_period():
    # main.pause_device() = source.stop() then source.start().
    rec = _Recorder(lambda n: _Cap())
    src = RtspFrameSource("rtsp://x", opener=rec, gate=_gate_ready())

    async def run():
        await src.start()
        assert _wait(lambda: rec.caps and rec.caps[0].i >= 3)
        await src.stop()
        assert rec.caps[0].released  # TEARDOWN path: the reader released it
        await src.start()
        assert _wait(lambda: len(rec.caps) >= 2)
        await src.stop()

    asyncio.run(run())
    assert rec.opened_at[1] - rec.caps[0].released_at >= Q * 0.95


def test_stop_during_quiet_wait_is_fast_and_opens_nothing():
    rec = _Recorder(lambda n: _Cap())
    src = RtspFrameSource("rtsp://x", opener=rec, reconnect_quiet_seconds=30)  # unknown boot: full wait

    async def run():
        await src.start()
        assert _wait(lambda: src.quiet_wait_remaining() > 25, timeout=1.0)
        t0 = time.monotonic()
        await src.stop()
        return time.monotonic() - t0

    took = asyncio.run(run())
    assert took < 2.0 and rec.caps == []
    assert src.quiet_wait_remaining() == 0


class _WedgeCap(_Cap):
    def __init__(self, n, gate):
        super().__init__()
        self.n = n
        self.gate = gate
        self.wedged = threading.Event()

    def read(self):
        if self.i >= self.n:
            self.wedged.set()
            self.gate.wait()
            return False, None
        return super().read()


def test_watchdog_replacement_waits_the_quiet_period_without_a_second_restart():
    unwedge = threading.Event()
    rec = _Recorder(lambda n: _WedgeCap(2, unwedge) if n == 0 else _Cap())
    quiet = 0.4
    src = RtspFrameSource("rtsp://x", opener=rec, stall_seconds=0.1, gate=_gate_ready(quiet))

    async def run():
        await src.start()
        assert _wait(lambda: rec.caps[0].wedged.is_set())
        await src.read()
        time.sleep(0.15)
        t_abandon = time.monotonic()
        await src.read()  # watchdog abandons the wedged reader
        assert src.stall_restarts == 1
        # During the quiet wait the watchdog must not fire again.
        for _ in range(4):
            time.sleep(0.07)
            await src.read()
        assert src.stall_restarts == 1
        assert _wait(lambda: len(rec.caps) >= 2)
        unwedge.set()
        assert _wait(lambda: rec.caps[0].released)  # abandoned reader released on unblock
        await src.stop()
        return t_abandon

    t_abandon = asyncio.run(run())
    assert rec.opened_at[1] - t_abandon >= quiet * 0.9
    assert src.last_session_end_ts is not None


def test_stop_join_default_outlasts_a_blocked_open_or_read():
    src = RtspFrameSource("rtsp://x", opener=lambda _u: _Cap(), io_timeout_seconds=5.0)
    # open timeout is max(2 x read, 10 s) = 10 s; stop waits that + 2 s.
    assert src._join_timeout == pytest.approx(12.0)


# --- Corruption self-heal ----------------------------------------------------


def test_tracker_needs_persistence_not_a_burst():
    clock = Clock(0.0)
    tr = CorruptionTracker(120, 0.8, min_samples=5, clock=clock)
    for _ in range(10):  # 10 corrupt checks in 50 s: not long enough yet
        assert tr.record(True) is False
        clock.t += 5
    for _ in range(14):
        clock.t += 5
        if tr.record(True):
            break
    else:
        pytest.fail("persistent corruption never healed")
    assert clock.t >= 120


def test_tracker_mixed_or_clean_never_heals():
    clock = Clock(0.0)
    tr = CorruptionTracker(120, 0.8, clock=clock)
    for i in range(100):  # half corrupt for 500 s
        assert tr.record(i % 2 == 0) is False
        clock.t += 5
    tr.reset()
    for _ in range(100):
        assert tr.record(False) is False
        clock.t += 5
    assert tr.corrupt_rate() == 0.0


def _scene(seed=0, h=240, w=320):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    import cv2

    return cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)


def _smeared(seed=0):
    f = _scene(seed)
    f[60:] = f[59]  # rows copied down: the incident's streaks
    return f


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((240, 320, 3), np.uint8),  # night
        np.full((240, 320, 3), 40, np.uint8),  # dark, flat
        np.full((240, 320, 3), (200, 160, 120), np.uint8),  # flat sky
        _scene(3),  # textured clean scene
    ],
    ids=["black", "dark-flat", "flat-sky", "clean-scene"],
)
def test_dark_flat_and_clean_frames_are_not_corrupt(frame):
    assert frame_report(frame)["corrupt"] is False


def test_smeared_frame_is_corrupt():
    assert frame_report(_smeared())["corrupt"] is True


def _heal_source(frames_factory, *, window=0.3, quiet=0.25, real_scorer=False):
    rec = _Recorder(frames_factory)
    kw = {} if real_scorer else {"frame_scorer": lambda f: {"corrupt": int(f[0, 0, 0]) == 7}}
    src = RtspFrameSource(
        "rtsp://x",
        opener=rec,
        gate=_gate_ready(quiet),
        heal_enabled=True,
        heal_check_every=1,
        heal_window_seconds=window,
        heal_min_share=0.8,
        **kw,
    )
    src._tracker.min_samples = 3
    return src, rec


def test_persistent_corruption_heals_release_quiet_reconnect():
    # Session 0 streams corrupt frames (value 7); session 1 is clean.
    src, rec = _heal_source(lambda n: _Cap(value=7) if n == 0 else _Cap(value=1))

    async def run():
        await src.start()
        t_end = time.monotonic() + 3
        while src.corruption_heals == 0 and time.monotonic() < t_end:
            await src.read()
            await asyncio.sleep(0.03)
        assert src.corruption_heals == 1
        t_heal = time.monotonic()
        assert src.last_error and "corrupt" in src.last_error
        assert _wait(lambda: rec.caps[0].released)  # old session torn down
        assert _wait(lambda: len(rec.caps) >= 2)
        return t_heal

    async def go():
        t_heal = await run()
        # Clean frames afterwards: no further heal.
        t_end = time.monotonic() + 0.8
        while time.monotonic() < t_end:
            await src.read()
            await asyncio.sleep(0.03)
        await src.stop()
        return t_heal

    t_heal = asyncio.run(go())
    # t_heal is taken up to one loop tick after the heal, so it only bounds
    # from above; the release -> reopen gap is the real quiet-period check.
    assert rec.opened_at[1] > t_heal
    assert rec.opened_at[1] - rec.caps[0].released_at >= 0.25 * 0.9
    assert src.corruption_heals == 1
    assert src.corrupt_frames >= 3 and src.corruption_checks > src.corrupt_frames


def test_heal_in_a_row_doubles_the_quiet_wait():
    src, rec = _heal_source(lambda n: _Cap(value=7))  # never clears
    src._gate.quiet_seconds = 10.0

    src._tracker._since = 0  # pretend a full window passed
    src._consecutive_heals = 0
    src._thread = None
    src._heal()
    first = src._gate.remaining()
    src._heal()
    second = src._gate.remaining()
    src._heal()
    third = src._gate.remaining()
    asyncio.run(src.stop())
    assert first == pytest.approx(10, abs=0.5)
    assert second == pytest.approx(20, abs=0.5)
    assert third == pytest.approx(40, abs=0.5)
    assert src.corruption_heals == 3


def test_clean_or_dark_streams_never_heal_with_the_real_scorer():
    dark = lambda n: _Cap(value=0)  # noqa: E731 -- black night frames
    src, rec = _heal_source(dark, real_scorer=True)

    async def run():
        await src.start()
        t_end = time.monotonic() + 0.8
        while time.monotonic() < t_end:
            await src.read()
            await asyncio.sleep(0.02)
        await src.stop()

    asyncio.run(run())
    assert src.corruption_checks >= 5
    assert src.corruption_heals == 0 and len(rec.caps) == 1


def test_heal_disabled_does_not_score():
    calls = []
    rec = _Recorder(lambda n: _Cap())
    src = RtspFrameSource(
        "rtsp://x", opener=rec, gate=_gate_ready(), heal_enabled=False, frame_scorer=lambda f: calls.append(1)
    )

    async def run():
        await src.start()
        assert _wait(lambda: src._latest is not None)
        await src.read()
        await src.stop()

    asyncio.run(run())
    assert calls == [] and src.corruption_checks == 0


# --- Settings / compose / env contract ---------------------------------------


def test_settings_defaults_are_on_and_state_path_is_per_stream(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("RETINA_"):
            monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None, RETINA_STREAM_ID="walk way/1")
    assert s.RETINA_RTSP_RECONNECT_QUIET_SECONDS == 90.0
    assert s.RETINA_RTSP_HEAL_ENABLED is True
    assert s.rtsp_session_state_path() == "/var/lib/orion-vision-retina/rtsp_session_walk_way_1.json"
    assert Settings(_env_file=None, RETINA_RTSP_SESSION_STATE_DIR="").rtsp_session_state_path() is None
    with pytest.raises(ValueError):
        Settings(_env_file=None, RETINA_RTSP_HEAL_MIN_SHARE=0)


def test_factory_threads_quiet_and_heal_settings(tmp_path):
    src = create_frame_source(
        "rtsp",
        "rtsp://x",
        rtsp_reconnect_quiet_seconds=90,
        rtsp_session_state_path=str(tmp_path / "s.json"),
        rtsp_heal_enabled=True,
        rtsp_heal_check_every=7,
        rtsp_heal_window_seconds=60,
        rtsp_heal_min_share=0.9,
    )
    assert src._gate.quiet_seconds == 90 and src._gate.state_path == str(tmp_path / "s.json")
    assert src.heal_enabled and src._heal_check_every == 7
    assert src._tracker.window_seconds == 60 and src._tracker.min_share == 0.9


KEYS = (
    "RETINA_RTSP_RECONNECT_QUIET_SECONDS",
    "RETINA_RTSP_SESSION_STATE_DIR",
    "RETINA_RTSP_HEAL_ENABLED",
    "RETINA_RTSP_HEAL_CHECK_EVERY",
    "RETINA_RTSP_HEAL_WINDOW_SECONDS",
    "RETINA_RTSP_HEAL_MIN_SHARE",
)


def test_both_compose_services_pass_the_keys_mount_state_and_allow_teardown():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    example = (ROOT / ".env_example").read_text()
    for name in ("orion-vision-retina", "orion-vision-retina-walkway"):
        svc = compose["services"][name]
        env = dict(e.split("=", 1) for e in svc["environment"])
        for key in KEYS:
            assert env[key].startswith("${" + key + ":-"), (name, key)
        assert env["RETINA_RTSP_HEAL_ENABLED"] == "${RETINA_RTSP_HEAL_ENABLED:-true}"
        assert "/mnt/telemetry/vision/retina_state:/var/lib/orion-vision-retina" in svc["volumes"]
        assert int(svc["stop_grace_period"].rstrip("s")) >= 15
    for key in KEYS:
        assert f"\n{key}=" in example
    assert "\nRETINA_RTSP_HEAL_ENABLED=true" in example
    assert "\nRETINA_RTSP_RECONNECT_QUIET_SECONDS=90.0" in example


# --- Service wiring: quiet wait is not a failed read; health carries metrics --


def _service_with(source):
    import sys

    repo = str(ROOT.parents[1])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from app.main import RetinaService

    class _Bus:
        async def publish(self, *_a, **_k):
            return None

    svc = RetinaService(settings=Settings(_env_file=None, RETINA_SOURCE_TYPE="folder"), bus=_Bus())
    svc.source = source
    return svc


class _QuietSource:
    stall_restarts = 0
    ever_connected = True
    last_error = None
    stall_error = None
    needs_process_restart = False
    last_session_end_ts = 123.0
    corruption_checks = 9
    corrupt_frames = 4
    corruption_heals = 2

    def __init__(self, remaining):
        self.remaining = remaining

    def quiet_wait_remaining(self):
        return self.remaining

    def corrupt_frame_rate(self):
        return 0.25

    async def read(self):
        return None


def test_capture_once_reports_the_quiet_wait_not_a_failed_read():
    svc = _service_with(_QuietSource(42.0))
    assert asyncio.run(svc.capture_once()) is False
    assert svc.metrics.frames_failed == 0
    assert "quiet period: 42s" in svc.metrics.last_error
    m = svc.metrics
    assert (m.source_quiet_wait_remaining_sec, m.source_last_session_end_ts) == (42.0, 123.0)
    assert (m.source_corruption_checks, m.source_corrupt_frames, m.source_corruption_heals) == (9, 4, 2)
    assert m.source_corrupt_frame_rate == 0.25


def test_capture_once_counts_a_real_miss_once_the_wait_is_over():
    svc = _service_with(_QuietSource(0.0))
    asyncio.run(svc.capture_once())
    assert svc.metrics.frames_failed == 1


def test_health_details_carry_session_and_heal_metrics():
    from app.health import RetinaMetrics, make_system_health_envelope

    m = RetinaMetrics(
        source_quiet_wait_remaining_sec=12.5,
        source_last_session_end_ts=99.0,
        source_corrupt_frame_rate=0.9,
        source_corruption_checks=10,
        source_corrupt_frames=9,
        source_corruption_heals=1,
    )
    env = make_system_health_envelope(
        service_name="s", service_version="v", camera_id="c", stream_id="walkway", source_type="rtsp",
        source_ok=False, metrics=m, fps_target=2.0, storage_dir="/x", heartbeat_interval_sec=10.0,
    )
    d = env.payload["details"]
    assert d["source_quiet_wait_remaining_sec"] == 12.5
    assert d["source_last_session_end_ts"] == 99.0
    assert d["source_corrupt_frame_rate"] == 0.9
    assert (d["source_corruption_checks"], d["source_corrupt_frames"], d["source_corruption_heals"]) == (10, 9, 1)
