from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload

from app.policy import FrameDispatchPolicy, load_policy_file
from app.settings import Settings
from app.state import RouterState


@pytest.fixture
def policy_path(tmp_path: Path) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 2
  min_seconds_between_tasks_per_camera: 0
  max_inflight_per_camera: 1
  request: {}
global:
  max_inflight_total: 2
  drop_when_busy: true
  require_image_path_exists: false
cameras:
  porch_eye:
    enabled: true
    task_type: detect_open_vocab
    every_n_frames: 1
    min_seconds_between_tasks_per_camera: 0
    request:
      prompts: [person, package]
  kitchen_eye:
    enabled: false
""",
        encoding="utf-8",
    )
    return p


def _frame_env(camera_id: str = "cam-a", image_path: str = "/tmp/f.jpg") -> BaseEnvelope:
    payload = VisionFramePointerPayload(
        image_path=image_path,
        camera_id=camera_id,
        frame_ts=time.time(),
    )
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-retina", version="0.1.0"),
        correlation_id=uuid4(),
        payload=payload.model_dump(mode="json"),
    )


def test_every_nth_frame_sampling(policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    env = _frame_env("cam-a")
    d1 = policy.decide(env, state, now=1.0)
    assert d1.should_dispatch is False
    assert d1.reason == "frame_sampled_out"
    d2 = policy.decide(env, state, now=2.0)
    assert d2.should_dispatch is True
    assert d2.task_type == "retina_fast"


def test_disabled_camera_skips(policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(_frame_env("kitchen_eye"), state, now=1.0)
    assert decision.should_dispatch is False
    assert decision.reason == "camera_disabled"


def test_camera_override_task_type(policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(_frame_env("porch_eye"), state, now=1.0)
    assert decision.should_dispatch is True
    assert decision.task_type == "detect_open_vocab"
    assert decision.request_overrides.get("prompts") == ["person", "package"]


def test_global_inflight_limit(tmp_path: Path) -> None:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 1
  min_seconds_between_tasks_per_camera: 0
  max_inflight_per_camera: 1
  request: {}
global:
  max_inflight_total: 1
  drop_when_busy: true
  require_image_path_exists: false
cameras: {}
""",
        encoding="utf-8",
    )
    settings = Settings(
        ROUTER_POLICY_PATH=str(p),
        REQUIRE_IMAGE_PATH_EXISTS=False,
        MAX_INFLIGHT_TOTAL=1,
    )
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    state.mark_dispatched(
        correlation_id="x",
        camera_id="cam-a",
        image_path="/tmp/a.jpg",
        task_type="retina_fast",
        reply_to="orion:vision:reply:x",
        now=1.0,
        frame_ts=None,
    )
    decision = policy.decide(_frame_env("cam-b"), state, now=2.0)
    assert decision.should_dispatch is False
    assert decision.reason == "global_inflight_limit"


def test_missing_image_path_skips(policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(policy_path), REQUIRE_IMAGE_PATH_EXISTS=True)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(_frame_env("cam-a", image_path=""), state, now=1.0)
    assert decision.should_dispatch is False
    assert decision.reason == "missing_image_path"


def test_image_path_not_visible(tmp_path: Path) -> None:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 1
  min_seconds_between_tasks_per_camera: 0
  max_inflight_per_camera: 1
  request: {}
global:
  max_inflight_total: 2
  require_image_path_exists: true
cameras: {}
""",
        encoding="utf-8",
    )
    settings = Settings(ROUTER_POLICY_PATH=str(p), REQUIRE_IMAGE_PATH_EXISTS=True)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(
        _frame_env("cam-a", image_path="/nonexistent/frame.jpg"),
        state,
        now=1.0,
        image_path_exists=False,
    )
    assert decision.should_dispatch is False
    assert decision.reason == "image_path_not_visible"


def test_camera_rate_limited(tmp_path: Path) -> None:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 1
  min_seconds_between_tasks_per_camera: 60
  max_inflight_per_camera: 2
  request: {}
global:
  max_inflight_total: 4
  drop_when_busy: true
  require_image_path_exists: false
cameras: {}
""",
        encoding="utf-8",
    )
    policy = FrameDispatchPolicy.load(Settings(ROUTER_POLICY_PATH=str(p)))
    state = RouterState()
    env = _frame_env("cam-a")
    first = policy.decide(env, state, now=1.0, image_path_exists=True)
    assert first.should_dispatch is True
    state.mark_dispatched(
        correlation_id="c1",
        camera_id="cam-a",
        image_path="/tmp/f.jpg",
        task_type="retina_fast",
        reply_to="orion:vision:reply:c1",
        now=1.0,
        frame_ts=None,
    )
    second = policy.decide(env, state, now=2.0, image_path_exists=True)
    assert second.should_dispatch is False
    assert second.reason == "camera_rate_limited"


def test_camera_inflight_limit(tmp_path: Path) -> None:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 1
  min_seconds_between_tasks_per_camera: 0
  max_inflight_per_camera: 1
  request: {}
global:
  max_inflight_total: 4
  require_image_path_exists: false
cameras: {}
""",
        encoding="utf-8",
    )
    policy = FrameDispatchPolicy.load(Settings(ROUTER_POLICY_PATH=str(p)))
    state = RouterState()
    state.mark_dispatched(
        correlation_id="c1",
        camera_id="cam-a",
        image_path="/tmp/f.jpg",
        task_type="retina_fast",
        reply_to="orion:vision:reply:c1",
        now=1.0,
        frame_ts=None,
    )
    decision = policy.decide(_frame_env("cam-a"), state, now=2.0, image_path_exists=True)
    assert decision.should_dispatch is False
    assert decision.reason == "camera_inflight_limit"


def test_router_disabled(policy_path: Path) -> None:
    settings = Settings(
        ROUTER_POLICY_PATH=str(policy_path),
        REQUIRE_IMAGE_PATH_EXISTS=False,
        ROUTER_ENABLED=False,
    )
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(_frame_env("cam-a"), state, now=1.0)
    assert decision.should_dispatch is False
    assert decision.reason == "router_disabled"


def test_drop_when_busy_false_uses_backpressure_reason(tmp_path: Path) -> None:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 1
  min_seconds_between_tasks_per_camera: 0
  max_inflight_per_camera: 2
  request: {}
global:
  max_inflight_total: 1
  drop_when_busy: false
  require_image_path_exists: false
cameras: {}
""",
        encoding="utf-8",
    )
    policy = FrameDispatchPolicy.load(Settings(ROUTER_POLICY_PATH=str(p)))
    state = RouterState()
    state.mark_dispatched(
        correlation_id="x",
        camera_id="cam-a",
        image_path="/tmp/a.jpg",
        task_type="retina_fast",
        reply_to="orion:vision:reply:x",
        now=1.0,
        frame_ts=None,
    )
    decision = policy.decide(_frame_env("cam-b"), state, now=2.0)
    assert decision.should_dispatch is False
    assert decision.reason == "global_inflight_backpressure"
