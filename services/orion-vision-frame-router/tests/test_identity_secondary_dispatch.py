"""dispatcher.py's secondary identity_face dispatch -- the actual bus
publish + RouterState bookkeeping around policy.decide_identity (see
test_decide_identity.py for the policy decision itself, unit-tested in
isolation). 2026-08-26, docs/superpowers/specs/2026-08-21-seeing-juniper-
identity-and-situated-observation-design.md sections 4/6.1.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload

from app.dispatcher import FrameDispatcher
from app.metrics import RouterMetrics
from app.policy import FrameDispatchPolicy
from app.settings import Settings
from app.state import RouterState


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel: str, envelope: object) -> None:
        self.published.append((channel, envelope))


@pytest.fixture
def identity_policy_path(tmp_path: Path) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  baseline:
    task_type: retina_fast
    every_n_frames: 1
    min_seconds_between_tasks_per_camera: 0
    request: {}
  triggered:
    task_type: retina_fast
    trigger_labels: [person]
    trigger_ttl_seconds: 8
    min_seconds_between_tasks_per_camera: 0
    max_inflight_per_camera: 2
    request:
      want_caption: true
global:
  max_inflight_total: 4
  require_image_path_exists: false
streams:
  cam0:
    enabled: true
    triggered:
      task_type: retina_fast
      trigger_labels: [person]
      trigger_ttl_seconds: 8
      min_seconds_between_tasks_per_camera: 0
      max_inflight_per_camera: 2
      request:
        want_caption: true
      identity_dispatch:
        enabled: true
        min_seconds_between_dispatch: 30
cameras: {}
""",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def live_identity_policy_path(tmp_path: Path) -> Path:
    """max_inflight_per_camera: 1 -- the REAL production value (config/
    vision_frame_router.yaml's live cam0 entry), not the 2 the fixture
    above uses. Review finding, 2026-08-26 (found independently by three
    separate review passes): the earlier fixture's max_inflight_per_camera
    of 2 masked the actual bug -- identity's own mark_dispatched call was
    consuming the SAME per-camera inflight slot the primary tier's
    decide() checks, so at the real value of 1, identity firing froze
    cam0's primary retina_fast dispatch until identity's own reply
    cleared. This fixture exists specifically so a regression here is
    caught by fixture shape, not just by reasoning about the code."""
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  baseline:
    task_type: retina_fast
    every_n_frames: 1
    min_seconds_between_tasks_per_camera: 0
    request: {}
  triggered:
    task_type: retina_fast
    trigger_labels: [person]
    trigger_ttl_seconds: 8
    min_seconds_between_tasks_per_camera: 0
    max_inflight_per_camera: 1
    request:
      want_caption: true
global:
  max_inflight_total: 4
  require_image_path_exists: false
streams:
  cam0:
    enabled: true
    triggered:
      task_type: retina_fast
      trigger_labels: [person]
      trigger_ttl_seconds: 8
      min_seconds_between_tasks_per_camera: 0
      max_inflight_per_camera: 1
      request:
        want_caption: true
      identity_dispatch:
        enabled: true
        min_seconds_between_dispatch: 30
cameras: {}
""",
        encoding="utf-8",
    )
    return p


def _make_dispatcher(policy_path: Path, *, dry_run: bool = False) -> tuple[FrameDispatcher, FakeBus]:
    settings = Settings(ROUTER_POLICY_PATH=str(policy_path), REQUIRE_IMAGE_PATH_EXISTS=False, DRY_RUN=dry_run)
    bus = FakeBus()
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    dispatcher = FrameDispatcher(settings=settings, policy=policy, state=state, metrics=RouterMetrics(), bus=bus)
    return dispatcher, bus


def _frame_env(*, correlation_id=None) -> BaseEnvelope:
    payload = VisionFramePointerPayload(
        image_path="/tmp/f.jpg",
        camera_id="rtsp://cam",
        stream_id="cam0",
        frame_ts=time.time(),
    )
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-edge", version="0.2.0"),
        correlation_id=correlation_id or uuid4(),
        payload=payload.model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_identity_dispatch_fires_alongside_primary_when_triggered(identity_policy_path: Path) -> None:
    dispatcher, bus = _make_dispatcher(identity_policy_path)
    dispatcher.state.record_activity("cam0", ["person"], now=time.time())

    await dispatcher.handle_frame_envelope(_frame_env())

    assert len(bus.published) == 2
    task_types = sorted(env.payload["task_type"] for _, env in bus.published)
    assert task_types == ["identity_face", "retina_fast"]
    assert dispatcher.metrics.identity_dispatched_total == 1
    assert dispatcher.metrics.frames_dispatched_total == 1  # unchanged by the secondary path


@pytest.mark.asyncio
async def test_identity_dispatch_uses_an_independent_correlation_id(identity_policy_path: Path) -> None:
    """The real bug a shared corr_id would cause: RouterState.pending is
    keyed by corr_id, so a collision would silently overwrite the primary
    task's pending entry and both replies would land on the same
    reply_to channel."""
    dispatcher, bus = _make_dispatcher(identity_policy_path)
    dispatcher.state.record_activity("cam0", ["person"], now=time.time())

    await dispatcher.handle_frame_envelope(_frame_env())

    corr_ids = {str(env.correlation_id) for _, env in bus.published}
    reply_tos = {env.reply_to for _, env in bus.published}
    assert len(corr_ids) == 2, "primary and identity tasks must not share a correlation_id"
    assert len(reply_tos) == 2, "primary and identity tasks must not share a reply_to channel"
    assert dispatcher.state.inflight_total() == 2
    assert len(dispatcher.state.pending) == 2


@pytest.mark.asyncio
async def test_identity_dispatch_does_not_fire_on_baseline_tier(identity_policy_path: Path) -> None:
    dispatcher, bus = _make_dispatcher(identity_policy_path)
    # No record_activity -- stays on the baseline tier.

    await dispatcher.handle_frame_envelope(_frame_env())

    assert len(bus.published) == 1
    assert bus.published[0][1].payload["task_type"] == "retina_fast"
    assert dispatcher.metrics.identity_dispatched_total == 0


@pytest.mark.asyncio
async def test_identity_dispatch_rate_limited_across_consecutive_triggered_frames(
    identity_policy_path: Path,
) -> None:
    """Hand-computed: min_seconds_between_dispatch=30. Two triggered frames
    dispatched back-to-back must only fire identity once."""
    dispatcher, bus = _make_dispatcher(identity_policy_path)
    dispatcher.state.record_activity("cam0", ["person"], now=time.time())

    await dispatcher.handle_frame_envelope(_frame_env())
    await dispatcher.handle_frame_envelope(_frame_env())

    assert dispatcher.metrics.identity_dispatched_total == 1


@pytest.mark.asyncio
async def test_identity_dispatch_respects_dry_run(identity_policy_path: Path) -> None:
    """DRY_RUN must suppress the identity publish exactly like the primary
    one -- but bookkeeping (mark_dispatched, the rate-limit timestamp,
    metrics) still happens, matching the primary path's own DRY_RUN
    contract (state.inflight_total()==1 in test_dry_run_records_without_
    publish)."""
    dispatcher, bus = _make_dispatcher(identity_policy_path, dry_run=True)
    dispatcher.state.record_activity("cam0", ["person"], now=time.time())

    await dispatcher.handle_frame_envelope(_frame_env())

    assert bus.published == []
    assert dispatcher.metrics.identity_dispatched_total == 1
    assert dispatcher.state.inflight_total() == 2


@pytest.mark.asyncio
async def test_identity_dispatch_does_not_starve_primary_detection_at_live_inflight_cap(
    live_identity_policy_path: Path,
) -> None:
    """The actual bug three review passes found independently, 2026-08-26,
    reproduced at the REAL production max_inflight_per_camera value (1),
    not a masking fixture value of 2. Sequence: frame 1 dispatches primary
    + identity (identity's own reply has not arrived yet, still pending);
    the PRIMARY reply then arrives and clears; frame 2 (a new triggered
    frame) must still be able to dispatch its own primary task -- if
    identity's mark_dispatched call had consumed the camera's inflight
    slot (the bug), frame 2 would be skipped with camera_inflight_limit
    even though the only thing still inflight is identity_face, not
    retina_fast."""
    dispatcher, bus = _make_dispatcher(live_identity_policy_path)
    dispatcher.state.record_activity("cam0", ["person"], now=time.time())

    await dispatcher.handle_frame_envelope(_frame_env())
    assert len(bus.published) == 2  # primary + identity, both dispatched

    # Clear the PRIMARY task's own pending entry (its reply arrived) --
    # identity's corr_id stays pending, exactly the scenario the bug hit.
    primary_corr = next(
        str(env.correlation_id) for _, env in bus.published if env.payload["task_type"] == "retina_fast"
    )
    identity_corr = next(
        str(env.correlation_id) for _, env in bus.published if env.payload["task_type"] == "identity_face"
    )
    dispatcher.state.clear_pending(primary_corr, now=time.time())
    assert identity_corr in dispatcher.state.pending, "identity task should still be pending"
    assert len(dispatcher.state.camera("rtsp://cam").inflight) == 0, (
        "identity's corr_id must never have occupied the per-camera inflight slot"
    )

    dispatcher.state.record_activity("cam0", ["person"], now=time.time())
    await dispatcher.handle_frame_envelope(_frame_env())

    primary_dispatches = [env for _, env in bus.published if env.payload["task_type"] == "retina_fast"]
    assert len(primary_dispatches) == 2, (
        "primary retina_fast dispatch must not be blocked by identity's still-pending task"
    )


# --- Full-resolution still before the face check (2026-10-11) ---------------------------------

@pytest.fixture
def still_policy_path(identity_policy_path: Path, tmp_path: Path) -> Path:
    p = tmp_path / "still_policy.yaml"
    p.write_text(
        identity_policy_path.read_text(encoding="utf-8").replace(
            "        min_seconds_between_dispatch: 30\n",
            "        min_seconds_between_dispatch: 0\n        still_url: http://edge:7100/still\n        still_timeout_sec: 3\n",
        ),
        encoding="utf-8",
    )
    return p


async def _drain(dispatcher: FrameDispatcher) -> None:
    import asyncio

    while dispatcher._background:
        await asyncio.gather(*list(dispatcher._background))


def _identity_published(bus: FakeBus):
    return [env for _, env in bus.published if env.payload["task_type"] == "identity_face"]


@pytest.mark.asyncio
async def test_face_check_uses_the_full_resolution_still(still_policy_path: Path, monkeypatch) -> None:
    import app.dispatcher as d

    calls = []

    def fake_still(url, timeout):
        calls.append((url, timeout))
        return {"ok": True, "image_path": "/frames/still_1.jpg", "width": 2560, "height": 1920, "grab_ms": 2100}

    monkeypatch.setattr(d, "request_still", fake_still)
    dispatcher, bus = _make_dispatcher(still_policy_path)
    dispatcher.state.record_activity("cam0", ["person"], now=time.time())

    await dispatcher.handle_frame_envelope(_frame_env())
    await _drain(dispatcher)

    assert calls == [("http://edge:7100/still", 3.0)]
    [identity] = _identity_published(bus)
    assert identity.payload["request"]["image_path"] == "/frames/still_1.jpg"
    assert identity.payload["request"]["image_source"] == "hires_still"
    assert identity.payload["meta"]["still_size"] == [2560, 1920]
    assert dispatcher.metrics.identity_still_total == 1
    assert dispatcher.state.camera("cam0").identity_still_pending is False
    pend = [p for p in dispatcher.state.pending.values() if p.task_type == "identity_face"]
    assert pend[0].image_path == "/frames/still_1.jpg"


@pytest.mark.asyncio
async def test_failed_still_falls_back_to_the_stream_frame(still_policy_path: Path, monkeypatch) -> None:
    import app.dispatcher as d

    def broken(url, timeout):
        raise d.StillError("no_frame")

    monkeypatch.setattr(d, "request_still", broken)
    dispatcher, bus = _make_dispatcher(still_policy_path)
    dispatcher.state.record_activity("cam0", ["person"], now=time.time())

    await dispatcher.handle_frame_envelope(_frame_env())
    await _drain(dispatcher)

    [identity] = _identity_published(bus)
    assert identity.payload["request"]["image_path"] == "/tmp/f.jpg"
    assert identity.payload["request"]["image_source"] == "stream_frame"
    assert identity.payload["meta"]["still_error"] == "no_frame"
    assert dispatcher.metrics.identity_still_fallback_total == 1
    assert dispatcher.state.camera("cam0").identity_still_pending is False


@pytest.mark.asyncio
async def test_no_second_face_check_while_a_still_is_being_fetched(still_policy_path: Path, monkeypatch) -> None:
    """min_seconds_between_dispatch is 0 here, so only the pending flag stops a pile-up."""
    import asyncio
    import threading

    import app.dispatcher as d

    release = threading.Event()
    calls = []

    def slow_still(url, timeout):
        calls.append(url)
        release.wait(5)
        return {"ok": True, "image_path": "/frames/still_2.jpg"}

    monkeypatch.setattr(d, "request_still", slow_still)
    dispatcher, bus = _make_dispatcher(still_policy_path)
    dispatcher.state.record_activity("cam0", ["person"], now=time.time())

    await dispatcher.handle_frame_envelope(_frame_env())
    await asyncio.sleep(0.05)
    # The primary path keeps flowing while the still is fetched.
    await dispatcher.handle_frame_envelope(_frame_env())
    assert len(calls) == 1
    assert sum(1 for _, e in bus.published if e.payload["task_type"] == "retina_fast") == 2
    release.set()
    await _drain(dispatcher)
    assert len(_identity_published(bus)) == 1


def test_request_still_reports_the_capture_service_error(monkeypatch) -> None:
    import io
    import urllib.error

    import app.dispatcher as d

    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 502, "bad", {}, io.BytesIO(b'{"ok": false, "error": "no_frame"}'))

    monkeypatch.setattr(d.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(d.StillError, match="no_frame"):
        d.request_still("http://edge:7100/still", 1.0)
