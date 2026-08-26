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
