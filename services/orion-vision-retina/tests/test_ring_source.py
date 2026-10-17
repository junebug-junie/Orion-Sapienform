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
        src.on_event(_event(kind="ding"))  # inside the cooldown
        await asyncio.sleep(0)
        return src, started

    src, started = asyncio.run(run())
    assert started == ["motion"]
    assert (src.events_seen, src.events_ignored, src.last_event_kind) == (2, 1, "ding")


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
    assert got.source_meta == {"type": "ring", "ring_event_kind": "ding"}
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
    assert svc["profiles"] == ["doorbell"]
    env = dict(e.split("=", 1) for e in svc["environment"])
    assert env["RETINA_SOURCE_TYPE"] == "ring"
    assert env["RETINA_STREAM_ID"] == "doorbell" and env["RETINA_CAMERA_ID"] == "doorbell"
    assert env["SERVICE_NAME"] == "vision-retina-doorbell"
    assert env["RETINA_CLIP_ENABLED"] == "false"
    assert any(v.endswith(":/var/lib/orion-vision-retina") for v in svc["volumes"])
    assert env["RETINA_RING_STATE_DIR"].startswith("/var/lib/orion-vision-retina/")
    repo = ROOT.parents[1]
    router = yaml.safe_load((repo / "config" / "vision_frame_router.yaml").read_text())
    zones = yaml.safe_load((repo / "config" / "vision_zones.yaml").read_text())
    policy = router["streams"]["doorbell"]
    # A ~30-frame burst must not be thinned like the continuous walkway.
    assert policy["baseline"]["every_n_frames"] == 1
    assert policy["caption_dispatch"]["enabled"] is True
    # Between bursts no frames arrive: a head count would read "nobody".
    assert all(not z.get("presence") for z in zones["streams"]["doorbell"]["zones"])
