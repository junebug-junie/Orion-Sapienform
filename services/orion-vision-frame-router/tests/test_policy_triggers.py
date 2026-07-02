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
def trigger_policy_path(tmp_path: Path) -> Path:
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
    request:
      want_caption: false
      want_embeddings: false
  triggered:
    task_type: retina_fast
    trigger_labels: [person, motion]
    trigger_ttl_seconds: 8
    min_seconds_between_tasks_per_camera: 0
    request:
      want_caption: true
      want_embeddings: true
global:
  max_inflight_total: 4
  require_image_path_exists: false
streams:
  cam0:
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


def test_resolve_policy_by_stream_id_fallback(trigger_policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(trigger_policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    merged, name = policy.resolve_stream_policy("rtsp://unknown", "cam0")
    assert name == "cam0"
    assert merged["baseline"]["request"]["want_caption"] is False


def test_triggered_dispatch_when_person_active(trigger_policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(trigger_policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    state.record_activity("cam0", ["person"], now=100.0)
    decision = policy.decide(_frame_env(), state, now=100.5, image_path_exists=True)
    assert decision.should_dispatch is True
    assert decision.dispatch_tier == "triggered"
    assert decision.request_overrides.get("want_caption") is True


def test_baseline_dispatch_without_trigger(trigger_policy_path: Path) -> None:
    settings = Settings(ROUTER_POLICY_PATH=str(trigger_policy_path), REQUIRE_IMAGE_PATH_EXISTS=False)
    policy = FrameDispatchPolicy.load(settings)
    state = RouterState()
    decision = policy.decide(_frame_env(), state, now=100.0, image_path_exists=True)
    assert decision.should_dispatch is True
    assert decision.dispatch_tier == "baseline"
    assert decision.request_overrides.get("want_caption") is False
