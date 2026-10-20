"""Ring doorbell source (app/ring_source.py): event filtering, cooldown,
frame hand-off to capture_loop, idle-is-not-failure, credentials on disk.
No network: Ring, FCM and WebRTC are never touched."""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from app.ring_source import RingEventFrameSource, hardware_id, matches_device, token_updater
from app.settings import Settings
from app.sources import create_frame_source

ROOT = Path(__file__).resolve().parents[1]
DOORBELL = SimpleNamespace(device_api_id=4242, name="Front Door")


def _event(**kw):
    base = dict(doorbot_id=4242, kind="ding", is_update=False)
    base.update(kw)
    return SimpleNamespace(**base)


class _Frame:
    """aiortc/av VideoFrame stand-in."""

    def __init__(self, h=480, w=640):
        self.arr = np.zeros((h, w, 3), dtype=np.uint8)

    def to_ndarray(self, format):  # noqa: A002
        assert format == "bgr24"
        return self.arr


def _connected(tmp_path, **kw) -> RingEventFrameSource:
    src = RingEventFrameSource("Front Door", state_dir=str(tmp_path), **kw)
    src._device = DOORBELL
    src.ever_connected = True
    return src


def test_factory_builds_ring_source(tmp_path):
    src = create_frame_source("ring", "Front Door", ring_state_dir=str(tmp_path), ring_burst_seconds=7)
    assert isinstance(src, RingEventFrameSource)
    assert src.burst_seconds == 7.0


def test_matches_device_by_id_or_app_name():
    assert matches_device(DOORBELL, "4242")
    assert matches_device(DOORBELL, "front door")
    assert matches_device(DOORBELL, "")  # empty = first camera
    assert not matches_device(DOORBELL, "Back Yard")


def test_event_filtering_and_cooldown(tmp_path):
    async def run():
        src = _connected(tmp_path, cooldown_seconds=30)
        started = []

        async def fake_burst(kind):
            started.append(kind)
            src._last_burst_end = time.time()

        src._burst = fake_burst
        src.on_event(_event(doorbot_id=1))  # another device
        src.on_event(_event(is_update=True))  # Ring re-sending the same event
        src.on_event(_event(kind="on_demand"))  # not a configured kind
        assert src.events_seen == 0
        src.on_event(_event(kind="motion"))
        await asyncio.sleep(0)
        src.on_event(_event(kind="motion"))  # inside the cooldown: no new view
        await asyncio.sleep(0)
        src.on_event(_event(kind="ding"))  # inside the cooldown: a ring always opens one
        await asyncio.sleep(0)
        return src, started

    src, started = asyncio.run(run())
    assert started == ["motion", "ding"]
    assert (src.events_seen, src.events_ignored, src.last_event_kind) == (3, 1, "ding")
    assert (src.dings_total, src.motions_total) == (1, 2) and src.last_ding_ts is not None


def test_no_second_burst_while_one_runs(tmp_path):
    async def run():
        src = _connected(tmp_path, cooldown_seconds=0)
        gate = asyncio.Event()

        async def slow_burst(kind):
            await gate.wait()

        src._burst = slow_burst
        src.on_event(_event())
        await asyncio.sleep(0)
        src.on_event(_event(kind="motion"))
        gate.set()
        await asyncio.sleep(0)
        return src

    src = asyncio.run(run())
    assert src.events_ignored == 1


def test_read_serves_each_frame_once_and_idle_is_waiting(tmp_path):
    async def run():
        src = _connected(tmp_path)
        assert src.waiting_for_event and await src.read() is None
        src._event_kind = "ding"
        src._frame, src._frame_ts = _Frame(), time.time()
        assert not src.waiting_for_event
        got = await src.read()
        again = await src.read()
        return src, got, again

    src, got, again = asyncio.run(run())
    assert (got.width, got.height) == (640, 480)
    assert got.source_meta == {"type": "ring", "ring_event_kind": "ding", "ring_correlation_id": None}
    assert again is None and src.waiting_for_event


def test_unauthenticated_source_is_not_waiting(tmp_path):
    src = RingEventFrameSource("", state_dir=str(tmp_path))
    assert not src.waiting_for_event  # capture_once reports its last_error instead


def test_connect_without_token_reports_login_needed(tmp_path):
    src = RingEventFrameSource("", state_dir=str(tmp_path))
    assert asyncio.run(src._connect_once()) is False
    assert "ring_login" in src.last_error and not src.ever_connected


def test_credentials_are_owner_only_and_hardware_id_is_stable(tmp_path):
    token_updater(str(tmp_path))({"refresh_token": "r"})
    mode = stat.S_IMODE(os.stat(tmp_path / "ring_token.json").st_mode)
    assert mode == 0o600
    assert hardware_id(str(tmp_path)) == hardware_id(str(tmp_path))


def _service_with(source):
    repo = str(ROOT.parents[1])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from app.main import RetinaService

    class _Bus:
        published = []

        async def publish(self, *a, **_k):
            self.published.append(a)

    svc = RetinaService(settings=Settings(_env_file=None, RETINA_SOURCE_TYPE="folder"), bus=_Bus())
    svc.source = source
    return svc


def test_capture_once_between_events_is_not_a_failure(tmp_path):
    svc = _service_with(_connected(tmp_path))
    assert asyncio.run(svc.capture_once()) is False
    assert svc.metrics.frames_failed == 0 and svc.metrics.last_error is None


def test_capture_once_surfaces_a_failed_live_view(tmp_path):
    src = _connected(tmp_path)
    src.last_error = "ring ding live view delivered no frame in 15s"
    src.bursts_failed = 1
    svc = _service_with(src)
    asyncio.run(svc.capture_once())
    assert svc.metrics.frames_failed == 0
    assert svc.metrics.last_error == src.last_error
    assert svc.metrics.ring_bursts_failed == 1


def test_capture_once_publishes_a_burst_frame(tmp_path, monkeypatch):
    src = _connected(tmp_path)
    src._frame, src._frame_ts, src._event_kind = _Frame(), time.time(), "motion"
    svc = _service_with(src)
    monkeypatch.setattr(svc.settings, "FRAME_STORAGE_DIR", str(tmp_path / "frames"))
    monkeypatch.setattr(svc.settings, "RETINA_STREAM_ID", "doorbell")
    monkeypatch.setattr(svc.settings, "RETINA_CAMERA_ID", "doorbell")
    assert asyncio.run(svc.capture_once()) is True
    assert svc.metrics.frames_published == 1
    payload = svc.bus.published[-1][1].payload
    assert payload["stream_id"] == "doorbell" and payload["camera_id"] == "doorbell"


def test_settings_refuse_ring_with_no_event_kinds():
    with pytest.raises(ValueError, match="RETINA_RING_EVENT_KINDS"):
        Settings(_env_file=None, RETINA_SOURCE_TYPE="ring", RETINA_RING_EVENT_KINDS=" , ")
    assert Settings(_env_file=None).ring_event_kinds() == ("motion", "ding")


def test_doorbell_instance_matches_router_and_zones():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    svc = compose["services"]["orion-vision-retina-doorbell"]
    # No profile: the post-pull rebuild runs a plain `up` and never starts a
    # profile service that is not already running, so a profiled doorbell
    # never deployed itself (2026-10-17). Unauthenticated, it only waits.
    assert "profiles" not in svc
    env = dict(e.split("=", 1) for e in svc["environment"])
    assert env["RETINA_SOURCE_TYPE"] == "ring"
    assert env["RETINA_STREAM_ID"] == "doorbell" and env["RETINA_CAMERA_ID"] == "doorbell"
    assert env["SERVICE_NAME"] == "vision-retina-doorbell"
    assert env["RETINA_CLIP_ENABLED"] == "false"
    # Ring credentials on their own mount: the other retinas share retina_state.
    assert "/mnt/telemetry/vision/ring_state:/var/lib/orion-vision-ring" in svc["volumes"]
    assert not any("retina_state" in v for v in svc["volumes"])
    assert env["RETINA_RING_STATE_DIR"] == "/var/lib/orion-vision-ring"
    for other in ("orion-vision-retina", "orion-vision-retina-walkway"):
        assert not any("ring_state" in v for v in compose["services"][other].get("volumes", []))
    repo = ROOT.parents[1]
    router = yaml.safe_load((repo / "config" / "vision_frame_router.yaml").read_text())
    zones = yaml.safe_load((repo / "config" / "vision_zones.yaml").read_text())
    policy = router["streams"]["doorbell"]
    # A ~30-frame burst must not be thinned like the continuous walkway.
    assert policy["baseline"]["every_n_frames"] == 1
    assert policy["caption_dispatch"]["enabled"] is True
    # Between bursts no frames arrive: a head count would read "nobody".
    assert all(not z.get("presence") for z in zones["streams"]["doorbell"]["zones"])


# --- live-view burst path (fake aiortc + fake Ring device) -------------------

import types

import app.ring_source as rs


class _Track:
    kind = "video"

    def __init__(self, frames: int, end_with=None):
        self.left = frames
        self.end_with = end_with

    async def recv(self):
        if self.left > 0:
            self.left -= 1
            await asyncio.sleep(0.01)
            return _Frame()
        if self.end_with is not None:
            raise self.end_with
        await asyncio.sleep(3600)


class _PC:
    instances: list = []

    def __init__(self, *_a, **_k):
        self.handlers = {}
        self.closed = False
        self.localDescription = SimpleNamespace(sdp="v=0\r\no=- 77 2 IN IP4 127.0.0.1\r\n")
        self.track = _Track(3)
        self.candidates = []
        _PC.instances.append(self)

    def addTransceiver(self, *_a, **_k):
        pass

    def on(self, event):
        def deco(fn):
            self.handlers[event] = fn
            return fn

        return deco

    async def createOffer(self):
        return None

    async def setLocalDescription(self, _d):
        pass

    async def setRemoteDescription(self, _d):
        self.handlers["track"](self.track)

    async def addIceCandidate(self, c):
        self.candidates.append(c)

    async def close(self):
        self.closed = True


class _Device:
    device_api_id = 4242
    name = "Front Door"

    def __init__(self, behaviour="answer", candidates=()):
        self.behaviour = behaviour
        self.candidates = candidates
        self.opened: list[str] = []
        self.closed: list[str] = []

    def get_ice_servers(self):
        return ["stun:example:3478"]

    async def generate_async_webrtc_stream(self, offer, session_id, cb, *, keep_alive_timeout=None):
        self.opened.append(session_id)
        self.keep_alive_timeout = keep_alive_timeout
        if self.behaviour == "raise":
            raise RuntimeError("websocket refused")
        if self.behaviour == "answer":
            for c in self.candidates:
                cb(SimpleNamespace(answer=None, candidate=c, sdp_m_line_index=0, error_code=None, error_message=None))
            cb(SimpleNamespace(answer="v=0 answer", candidate=None, sdp_m_line_index=None, error_code=None, error_message=None))
        if self.behaviour == "error":
            cb(SimpleNamespace(answer=None, candidate=None, sdp_m_line_index=None, error_code="busy", error_message="in use"))

    async def close_webrtc_stream(self, session_id):
        self.closed.append(session_id)


@pytest.fixture
def fake_aiortc(monkeypatch):
    _PC.instances = []
    mod = types.ModuleType("aiortc")
    mod.RTCPeerConnection = _PC
    mod.RTCConfiguration = lambda iceServers: iceServers
    mod.RTCIceServer = lambda urls: urls
    mod.RTCSessionDescription = lambda sdp, type: (sdp, type)  # noqa: A002
    sdp_mod = types.ModuleType("aiortc.sdp")
    sdp_mod.candidate_from_sdp = lambda s: SimpleNamespace(sdp=s, sdpMLineIndex=None)
    monkeypatch.setitem(sys.modules, "aiortc", mod)
    monkeypatch.setitem(sys.modules, "aiortc.sdp", sdp_mod)
    monkeypatch.setattr(rs, "ANSWER_TIMEOUT_SECONDS", 0.05)
    return _PC


def _burst_src(tmp_path, device, burst=0.3):
    src = RingEventFrameSource("Front Door", state_dir=str(tmp_path), burst_seconds=burst, cooldown_seconds=0)
    src._device = device
    src.ever_connected = True
    return src


def test_burst_delivers_frames_and_always_closes_the_session(tmp_path, fake_aiortc):
    dev = _Device(candidates=["candidate:1 1 udp 1 1.2.3.4 5 typ host"])

    async def run():
        src = _burst_src(tmp_path, dev)
        seen = []
        task = asyncio.create_task(src._burst("ding"))
        while not task.done():
            r = await src.read()
            if r is not None:
                seen.append(r)
            await asyncio.sleep(0.005)
        return src, seen

    src, seen = asyncio.run(run())
    assert seen and src.bursts_failed == 0 and src.last_error is None
    assert dev.closed == dev.opened and len(dev.opened) == 1  # registered before Ring talks, closed after
    assert dev.keep_alive_timeout >= 30  # Ring keeps the view open for the whole burst
    pc = fake_aiortc.instances[0]
    assert pc.closed and pc.candidates  # pre-answer candidate applied after the answer
    assert src._frame is None and src.waiting_for_event


def test_no_answer_is_named_failed_and_still_closed(tmp_path, fake_aiortc):
    dev = _Device(behaviour="silent")
    src = _burst_src(tmp_path, dev)
    asyncio.run(src._burst("motion"))
    assert src.bursts_failed == 1 and "no answer from Ring" in src.last_error
    assert dev.closed == dev.opened and fake_aiortc.instances[0].closed


@pytest.mark.parametrize("behaviour,text", [("raise", "websocket refused"), ("error", "Ring refused the live view: busy")])
def test_ring_errors_are_failed_and_still_closed(tmp_path, fake_aiortc, behaviour, text):
    dev = _Device(behaviour=behaviour)
    src = _burst_src(tmp_path, dev)
    asyncio.run(src._burst("ding"))
    assert src.bursts_failed == 1 and text in src.last_error
    assert dev.closed == dev.opened


def test_stream_ending_early_after_frames_is_not_a_failure(tmp_path, fake_aiortc, monkeypatch):
    dev = _Device()
    orig = _PC.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.track = _Track(2, end_with=RuntimeError("MediaStreamError"))

    monkeypatch.setattr(_PC, "__init__", init)
    src = _burst_src(tmp_path, dev, burst=5)
    asyncio.run(src._burst("ding"))
    assert src.bursts_failed == 0 and src.last_error is None
    assert dev.closed == dev.opened


def test_stop_mid_burst_waits_for_cleanup(tmp_path, fake_aiortc):
    dev = _Device()

    async def run():
        src = _burst_src(tmp_path, dev, burst=60)
        src._burst_task = asyncio.create_task(src._burst("ding"))
        await asyncio.sleep(0.05)
        await src.stop()
        return src

    src = asyncio.run(run())
    assert dev.closed == dev.opened and fake_aiortc.instances[0].closed
    assert src._burst_task is None and not src.ever_connected and src._device is None


def test_listener_alive_trusts_the_receiver_not_the_started_flag():
    rx = SimpleNamespace(is_started=lambda: False)
    assert not rs.listener_alive(SimpleNamespace(started=True, _receiver=rx))
    assert rs.listener_alive(SimpleNamespace(started=True, _receiver=SimpleNamespace(is_started=lambda: True)))
    assert not rs.listener_alive(None)


def test_supervisor_reconnects_a_dead_push_listener(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "SUPERVISE_SECONDS", 0.01)

    async def run():
        src = _connected(tmp_path)
        src._connected_at = time.time()
        src._listener = SimpleNamespace(started=True, _receiver=SimpleNamespace(is_started=lambda: False))

        async def _stop():
            return None

        src._listener.stop = _stop
        reconnects = []

        async def fake_connect():
            reconnects.append(src.last_error)
            src.ever_connected = True
            src._connected_at = time.time()
            src._listener = SimpleNamespace(started=True, _receiver=SimpleNamespace(is_started=lambda: True))
            src.last_error = None
            return True

        src._connect_once = fake_connect
        task = asyncio.create_task(src._supervise())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return reconnects

    reconnects = asyncio.run(run())
    assert reconnects == ["ring push listener stopped; reconnecting"]


# --- every Ring push is recorded (2026-10-20 live bug) -----------------------


def test_ding_during_open_live_view_is_recorded_and_extends_it(tmp_path):
    """Walk up = motion burst; the press arrived while it ran and was dropped
    ("ignored: burst running or in cooldown"). Now it is counted, handed to the
    sink, and pushes the open live view out."""

    async def run():
        src = _connected(tmp_path, cooldown_seconds=30, burst_seconds=15)
        pushes = []
        src.event_sink = pushes.append
        gate = asyncio.Event()

        async def slow_burst(kind):
            await gate.wait()

        src._burst = slow_burst
        src.on_event(_event(kind="motion", id=1))
        await asyncio.sleep(0)
        before = time.monotonic()
        src.on_event(_event(kind="ding", id=2))
        extend_until = src._burst_extend_until
        gate.set()
        await asyncio.sleep(0)
        return src, pushes, before, extend_until

    src, pushes, before, extend_until = asyncio.run(run())
    assert src.dings_total == 1 and src.motions_total == 1 and src.bursts_extended == 1
    assert extend_until >= before + 15 - 0.5
    assert src.events_ignored == 0
    assert [(p.kind, p.live_view) for p in pushes] == [("motion", "started"), ("ding", "extended")]
    assert pushes[1].ring_event_id == "2" and pushes[0].correlation_id != pushes[1].correlation_id
    assert src._event_kind == "ding" and src._burst_correlation_id == pushes[1].correlation_id


@pytest.mark.parametrize("raw", ["ding", "DING", "com.ring.push.HANDLE_NEW_DING", "com.ring.pn.live-event.ding"])
def test_every_ding_spelling_ring_doorbell_produces_is_a_ding(tmp_path, raw):
    """ring_doorbell 0.9.14 eventlistener.py:228 compares `action.lower` (the
    method) so a legacy-format ding carries the raw action string as kind."""

    async def run():
        src = _connected(tmp_path)
        src._burst = lambda kind: asyncio.sleep(0)
        src.on_event(_event(kind=raw))
        await asyncio.sleep(0)
        return src

    src = asyncio.run(run())
    assert src.dings_total == 1 and src.last_event_kind == "ding"


def test_legacy_motion_spelling_is_a_motion(tmp_path):
    async def run():
        src = _connected(tmp_path)
        src._burst = lambda kind: asyncio.sleep(0)
        src.on_event(_event(kind="com.ring.push.HANDLE_NEW_motion"))
        await asyncio.sleep(0)
        return src

    assert asyncio.run(run()).motions_total == 1


def test_sink_failure_never_breaks_recording(tmp_path):
    async def run():
        src = _connected(tmp_path)
        src._burst = lambda kind: asyncio.sleep(0)

        def boom(_p):
            raise RuntimeError("bus down")

        src.event_sink = boom
        src.on_event(_event())
        await asyncio.sleep(0)
        return src

    assert asyncio.run(run()).dings_total == 1


def test_real_burst_is_extended_by_a_ding(tmp_path, fake_aiortc, monkeypatch):
    """The frame loop reads the extension: a 0.3 s burst runs past 0.3 s when
    a ding lands mid-burst, and stops at MAX_BURST_FACTOR bursts at most."""
    dev = _Device()
    orig = _PC.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.track = _Track(1000)

    monkeypatch.setattr(_PC, "__init__", init)

    async def run():
        src = _burst_src(tmp_path, dev, burst=0.3)
        t0 = time.monotonic()
        task = asyncio.create_task(src._burst("motion"))
        await asyncio.sleep(0.2)
        src._extend_burst("c-1")
        await task
        return src, time.monotonic() - t0

    src, took = asyncio.run(run())
    assert took >= 0.45, took
    assert took < 0.3 * rs.MAX_BURST_FACTOR + 0.3
    assert dev.keep_alive_timeout >= int(0.3 * rs.MAX_BURST_FACTOR) + 30
    assert dev.closed == dev.opened and src.bursts_failed == 0


# --- active looking ----------------------------------------------------------


def test_self_look_caps_spacing_and_open_views(tmp_path):
    async def run():
        src = _connected(tmp_path, cooldown_seconds=0, self_look_max_per_day=2, self_look_min_spacing_seconds=100)
        kinds = []

        async def fake_burst(kind):
            kinds.append(kind)
            src._last_burst_end = time.time()

        src._burst = fake_burst
        # 01:00 UTC tomorrow: later than the real clock the fake burst stamps
        # into _last_burst_end, and far from a UTC day boundary.
        t = (time.time() // 86400 + 1) * 86400 + 3600
        out = [src.self_look("h1", now=t)]
        await asyncio.sleep(0)
        out.append(src.self_look("h2", now=t + 10))  # spacing
        out.append(src.self_look("h2", now=t + 200))
        await asyncio.sleep(0)
        out.append(src.self_look("h3", now=t + 400))  # cap 2
        out.append(src.self_look("h4", now=t + 86400))  # next UTC day
        return src, out, kinds

    src, out, kinds = asyncio.run(run())
    assert out == ["started", "spacing", "started", "daily_cap", "started"]
    assert kinds == ["self_look"] * 3
    assert src.self_looks_total == 3 and src.self_looks_today == 1
    assert src.last_self_look_reason == "h4"


def test_self_look_never_opens_a_second_view_or_while_deaf(tmp_path):
    async def run():
        src = _connected(tmp_path, cooldown_seconds=30)
        gate = asyncio.Event()

        async def slow_burst(kind):
            await gate.wait()

        src._burst = slow_burst
        src.on_event(_event(kind="motion"))
        await asyncio.sleep(0)
        running = src.self_look("h1")
        gate.set()
        await asyncio.sleep(0)
        src._last_burst_end = time.time()
        cooling = src.self_look("h1")
        deaf = RingEventFrameSource("", state_dir=str(tmp_path)).self_look("h1")
        return src, running, cooling, deaf

    src, running, cooling, deaf = asyncio.run(run())
    assert (running, cooling, deaf) == ("burst_running", "cooldown", "not_ready")
    assert src.self_looks_total == 0


class _Redis:
    def __init__(self):
        self.value = None

    async def get(self, _key):
        return self.value


def test_service_acts_once_per_key_value(tmp_path):
    async def run():
        src = _connected(tmp_path, cooldown_seconds=0, self_look_min_spacing_seconds=0)
        src._burst = lambda kind: asyncio.sleep(0)
        svc = _service_with(src)
        svc.bus.redis = _Redis()
        out = [await svc.self_look_once()]  # no key
        svc.bus.redis.value = b"handoff-7"
        out.append(await svc.self_look_once())
        await asyncio.sleep(0)
        out.append(await svc.self_look_once())  # same value: nothing
        svc.bus.redis.value = None
        out.append(await svc.self_look_once())
        for _ in range(3):
            await asyncio.sleep(0)  # let the first fake burst finish
        svc.bus.redis.value = "handoff-7"  # SET again after expiry counts
        out.append(await svc.self_look_once())
        return src, out

    src, out = asyncio.run(run())
    assert out == [None, "started", None, None, "started"]
    assert src.self_looks_total == 2 and src.last_self_look_reason == "handoff-7"


# --- recorded events on the bus ------------------------------------------------


def _push(kind="ding", **kw):
    base = dict(
        kind=kind,
        raw_kind=kind,
        ring_event_id="9001",
        occurred_at=1_760_000_000.0,
        received_ts=1_760_000_001.0,
        correlation_id="5b2f7b8e-3a0d-4a43-9d51-2c7f3c0b1a11",
        live_view="extended",
    )
    base.update(kw)
    return rs.RingPush(**base)


def test_ring_push_publishes_vision_and_street_event(tmp_path):
    from orion.schemas.registry import resolve
    from orion.schemas.vision import StreetEventV1, VisionEventPayload

    svc = _service_with(_connected(tmp_path))
    svc.settings.RETINA_STREAM_ID = "doorbell"
    svc.bus.published = []
    asyncio.run(svc._publish_ring_push(_push()))
    chans = [a[0] for a in svc.bus.published]
    assert chans == ["orion:vision:events", "orion:vision:street:events"]
    vis_env, street_env = (a[1] for a in svc.bus.published)
    assert vis_env.kind == "vision.event.bundle" and street_env.kind == "vision.street.event.v1"
    assert str(vis_env.correlation_id) == str(street_env.correlation_id) == _push().correlation_id
    item = VisionEventPayload(**vis_env.payload).events[0]
    assert (item.event_type, item.salience, item.stream_id) == ("doorbell_ring", 1.0, "doorbell")
    assert "ring_event:9001" in item.evidence_refs
    street = resolve("StreetEventV1").model_validate(street_env.payload)
    assert isinstance(street, StreetEventV1)
    assert (street.kind, street.salience, street.stream_ids, street.event_id) == ("doorbell_ring", 1.0, ["doorbell"], item.event_id)
    assert svc.metrics.ring_vision_events_published == 1 and svc.metrics.ring_street_events_published == 1


def test_motion_push_is_a_low_salience_vision_event_only(tmp_path):
    from orion.schemas.vision import VisionEventPayload

    svc = _service_with(_connected(tmp_path))
    svc.bus.published = []
    asyncio.run(svc._publish_ring_push(_push(kind="motion")))
    assert [a[0] for a in svc.bus.published] == ["orion:vision:events"]
    item = VisionEventPayload(**svc.bus.published[0][1].payload).events[0]
    assert item.event_type == "doorbell_motion" and item.salience == 0.2


def test_ring_event_id_is_stable_per_push():
    from app.ring_events import ring_event_id

    assert ring_event_id(_push()) == ring_event_id(_push(correlation_id="00000000-0000-0000-0000-000000000000"))
    assert ring_event_id(_push()) != ring_event_id(_push(ring_event_id="9002"))


def test_health_reports_ring_counts(tmp_path):
    src = _connected(tmp_path)
    src.dings_total, src.motions_total, src.last_ding_ts = 2, 5, 123.0
    src.self_looks_today, src.self_looks_total = 1, 4
    svc = _service_with(src)
    svc._copy_source_metrics()
    d = svc.health_envelope(None).payload["details"]
    assert (d["dings_total"], d["motions_total"], d["last_ding_ts"]) == (2, 5, 123.0)
    assert (d["self_looks_today"], d["self_looks_total"]) == (1, 4)


def test_doorbell_compose_ships_self_look_on():
    doc = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    env = doc["services"]["orion-vision-retina-doorbell"]["environment"]
    assert "RETINA_RING_SELF_LOOK_ENABLED=${DOORBELL_RING_SELF_LOOK_ENABLED:-true}" in env
    assert Settings(_env_file=None).RETINA_RING_SELF_LOOK_ENABLED is True
