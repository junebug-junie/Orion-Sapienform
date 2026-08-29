"""policy.decide_identity -- the secondary, independent identity_face
dispatch decision (2026-08-26, docs/superpowers/specs/2026-08-21-seeing-
juniper-identity-and-situated-observation-design.md sections 4/6.1).
Deliberately NOT folded into decide()/FrameDispatchDecision.should_dispatch
-- see decide_identity's own docstring for why.
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

from app.policy import FrameDispatchPolicy
from app.settings import Settings
from app.state import RouterState


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
    max_inflight_per_camera: 1
    request: {}
global:
  max_inflight_total: 2
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
      request: {}
      identity_dispatch:
        enabled: true
        min_seconds_between_dispatch: 30
  carbon:
    enabled: true
cameras: {}
""",
        encoding="utf-8",
    )
    return p


def _frame_env(camera_id: str = "rtsp://cam", stream_id: str = "cam0") -> BaseEnvelope:
    payload = VisionFramePointerPayload(
        image_path="/tmp/f.jpg",
        camera_id=camera_id,
        stream_id=stream_id,
        frame_ts=time.time(),
    )
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-edge", version="0.2.0"),
        correlation_id=uuid4(),
        payload=payload.model_dump(mode="json"),
    )


def _policy(path: Path) -> FrameDispatchPolicy:
    settings = Settings(ROUTER_POLICY_PATH=str(path), REQUIRE_IMAGE_PATH_EXISTS=False)
    return FrameDispatchPolicy.load(settings)


def test_identity_dispatch_true_when_triggered_and_enabled(identity_policy_path: Path) -> None:
    policy = _policy(identity_policy_path)
    state = RouterState()
    state.record_activity("cam0", ["person"], now=100.0)
    decision = policy.decide(_frame_env(), state, now=100.5, image_path_exists=True)
    assert decision.dispatch_tier == "triggered"
    assert policy.decide_identity(decision, camera_id="rtsp://cam", state=state, now=100.5) is True


def test_identity_dispatch_false_on_baseline_tier(identity_policy_path: Path) -> None:
    """Never-triggered frames must not get a secondary identity dispatch,
    even if identity_dispatch were somehow enabled on baseline (it isn't
    here, but this also proves the dispatch_tier gate is load-bearing on
    its own)."""
    policy = _policy(identity_policy_path)
    state = RouterState()
    decision = policy.decide(_frame_env(), state, now=100.0, image_path_exists=True)
    assert decision.dispatch_tier == "baseline"
    assert policy.decide_identity(decision, camera_id="rtsp://cam", state=state, now=100.0) is False


def test_identity_dispatch_false_when_not_configured(identity_policy_path: Path) -> None:
    """carbon has no identity_dispatch block at all -- must default to
    False (opt-in, not opt-out). This is the design doc's §9B requirement:
    carbon must not inherit cam0's identity policy."""
    policy = _policy(identity_policy_path)
    state = RouterState()
    state.record_activity("carbon", ["person"], now=100.0)
    decision = policy.decide(
        _frame_env(camera_id="carbon-webcam", stream_id="carbon"), state, now=100.5, image_path_exists=True
    )
    assert decision.dispatch_tier == "triggered"
    assert policy.decide_identity(decision, camera_id="carbon-webcam", state=state, now=100.5) is False


def test_identity_dispatch_rate_limited_per_camera(identity_policy_path: Path) -> None:
    """Hand-computed: min_seconds_between_dispatch=30. Dispatched at t=100,
    re-checked at t=110 (10s later, < 30) -- must refuse; at t=131 (31s
    later) -- must allow."""
    policy = _policy(identity_policy_path)
    state = RouterState()
    state.record_activity("cam0", ["person"], now=100.0)
    decision = policy.decide(_frame_env(), state, now=100.5, image_path_exists=True)

    state.camera("rtsp://cam").last_identity_dispatch_ts = 100.0
    assert policy.decide_identity(decision, camera_id="rtsp://cam", state=state, now=110.0) is False
    assert policy.decide_identity(decision, camera_id="rtsp://cam", state=state, now=131.0) is True


def test_identity_dispatch_respects_global_inflight_cap(identity_policy_path: Path) -> None:
    """max_inflight_total=2 in the fixture. Two pending tasks already
    occupy it -- identity must not push a third dispatch through even
    though max_inflight_per_camera (checked only by decide(), not
    decide_identity) would allow it."""
    policy = _policy(identity_policy_path)
    state = RouterState()
    state.record_activity("cam0", ["person"], now=100.0)
    decision = policy.decide(_frame_env(), state, now=100.5, image_path_exists=True)

    state.mark_dispatched(
        correlation_id="c1", camera_id="rtsp://cam", image_path="/tmp/f.jpg",
        task_type="retina_fast", reply_to="r1", now=100.5, frame_ts=None,
    )
    state.mark_dispatched(
        correlation_id="c2", camera_id="rtsp://cam", image_path="/tmp/f.jpg",
        task_type="retina_fast", reply_to="r2", now=100.5, frame_ts=None,
    )
    assert state.inflight_total() == 2
    assert policy.decide_identity(decision, camera_id="rtsp://cam", state=state, now=100.5) is False


def test_identity_dispatch_not_gated_by_max_inflight_per_camera(identity_policy_path: Path) -> None:
    """The whole point of NOT reusing max_inflight_per_camera (live: 1):
    the primary task for THIS SAME frame already occupies that slot by the
    time decide_identity runs. Reusing it would make the feature
    permanently inert on exactly the config it targets. global inflight
    (2) still has room, so this must be allowed."""
    policy = _policy(identity_policy_path)
    state = RouterState()
    state.record_activity("cam0", ["person"], now=100.0)
    decision = policy.decide(_frame_env(), state, now=100.5, image_path_exists=True)

    # Simulate the primary task's own mark_dispatched already having run --
    # exactly what dispatcher.py does before calling decide_identity.
    state.mark_dispatched(
        correlation_id="c1", camera_id="rtsp://cam", image_path="/tmp/f.jpg",
        task_type="retina_fast", reply_to="r1", now=100.5, frame_ts=None,
    )
    assert len(state.camera("rtsp://cam").inflight) == 1  # at max_inflight_per_camera already
    assert policy.decide_identity(decision, camera_id="rtsp://cam", state=state, now=100.5) is True


# -- the REAL config, not a fixture, 2026-08-29 ------------------------------
# Every test above builds its own synthetic policy, which is exactly how this
# feature could be "fully tested" and still never once run on the camera
# pointed at Juniper's face: the fixture said carbon was opted in for the
# negative case, and nothing asserted anything about the shipped file. Live
# evidence 2026-08-29: 2 identity_face dispatches in 24h, both stream_id=cam0.
# These tests read config/vision_frame_router.yaml itself.

REAL_POLICY = Path(__file__).resolve().parents[3] / "config" / "vision_frame_router.yaml"


def _real_policy() -> FrameDispatchPolicy:
    assert REAL_POLICY.exists(), f"shipped policy missing at {REAL_POLICY}"
    return _policy(REAL_POLICY)


@pytest.mark.parametrize(
    "camera_id,stream_id",
    [("carbon-webcam", "carbon"), ("rtsp://admin@192.168.1.21:554/Preview_01_sub", "cam0")],
)
def test_shipped_config_dispatches_identity_for_both_real_cameras(camera_id, stream_id) -> None:
    """carbon is Juniper's laptop webcam and cam0 the interior room camera.
    Whichever one she is actually in front of has to be able to answer 'is
    this Juniper', or the ask downstream can only ever fire on the other."""
    policy = _real_policy()
    state = RouterState()
    state.record_activity(stream_id, ["person"], now=100.0)
    # The shipped config samples 1-in-10 frames (DEFAULT_EVERY_N_FRAMES=10),
    # unlike the synthetic fixtures above which dispatch on every frame. Drive
    # real frames until one is actually sampled, the way the live router does
    # -- asserting on a single hand-picked frame would just be asserting where
    # the sampler's modulo happens to land.
    decision = None
    for _ in range(30):
        candidate = policy.decide(
            _frame_env(camera_id=camera_id, stream_id=stream_id),
            state,
            now=100.5,
            image_path_exists=True,
        )
        if candidate.dispatch_tier == "triggered" and candidate.should_dispatch:
            decision = candidate
            break
    assert decision is not None, "no frame was sampled in 30 tries"
    assert decision.identity_dispatch_cfg.get("enabled") is True
    assert policy.decide_identity(decision, camera_id=camera_id, state=state, now=100.5) is True


def test_shipped_config_keeps_identity_opt_in_for_other_streams() -> None:
    """Still opt-in, not a defaults change -- the design doc's §9B concern was
    a permissive global default silently reaching every camera, and that
    reasoning survives carbon being added explicitly."""
    policy = _real_policy()
    state = RouterState()
    state.record_activity("kitchen", ["person"], now=100.0)
    dispatched = None
    for _ in range(30):
        decision = policy.decide(
            _frame_env(camera_id="kitchen-cam", stream_id="kitchen"),
            state,
            now=100.5,
            image_path_exists=True,
        )
        if decision.dispatch_tier == "triggered" and decision.should_dispatch:
            dispatched = decision
            break
    # Without this, the test passes vacuously: if nothing is ever sampled,
    # decide_identity returns False because should_dispatch is False, not
    # because kitchen is opted out -- and both assertions below still hold
    # (review finding, 2026-08-29).
    assert dispatched is not None, "no frame was sampled in 30 tries"
    decision = dispatched
    assert decision.dispatch_tier == "triggered"
    assert decision.identity_dispatch_cfg.get("enabled") is not True
    assert policy.decide_identity(decision, camera_id="kitchen-cam", state=state, now=100.5) is False


def test_shipped_carbon_block_keeps_the_fields_the_shallow_merge_would_drop() -> None:
    """resolve_stream_policy REPLACES the whole `triggered` key rather than
    merging per field, so an override that omits trigger_labels would silently
    stop carbon ever reaching the triggered tier at all -- and identity would
    be dead again, this time invisibly."""
    policy = _real_policy()
    triggered = policy.streams_cfg["carbon"]["triggered"]
    assert triggered["trigger_labels"] == ["person"]
    assert triggered["task_type"] == "retina_fast"
    assert triggered["trigger_ttl_seconds"] == 8
    assert triggered["request"]["want_caption"] is True
