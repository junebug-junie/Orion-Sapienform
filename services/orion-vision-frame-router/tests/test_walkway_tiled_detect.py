"""Walkway tiled detection gate (2026-10-15): only the walkway's primary
tasks get request.detect_tiles, and at most one per min_seconds_between."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload

from app.policy import FrameDispatchPolicy
from app.settings import Settings
from app.state import RouterState

CFG = Path(__file__).resolve().parents[3] / "config" / "vision_frame_router.yaml"


def _policy() -> FrameDispatchPolicy:
    return FrameDispatchPolicy.load(Settings(ROUTER_POLICY_PATH=str(CFG)))


def _env(stream_id: str) -> BaseEnvelope:
    payload = VisionFramePointerPayload(
        image_path="/tmp/f.jpg", camera_id=stream_id, stream_id=stream_id, frame_ts=time.time()
    )
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-edge", version="0.2.0"),
        correlation_id=uuid4(),
        payload=payload.model_dump(mode="json"),
    )


def _task(policy, state, stream_id: str, now: float):
    """Build the primary task for one frame the way the dispatcher does,
    including the tiled_detect gate. Returns (task, tiled) or (None, False)
    when the frame is sampled out or rate-limited."""
    env = _env(stream_id)
    for _ in range(20):  # every_n_frames sampling: feed frames until one is taken
        d = policy.decide(env, state, now=now, image_path_exists=True)
        if d.should_dispatch or d.reason != "frame_sampled_out":
            break
    if not d.should_dispatch:
        return None, False
    frame = VisionFramePointerPayload.model_validate(env.payload)
    task = policy.build_task_request(frame, env, d)
    tiled = policy.apply_tiled_detect(task, frame, d, state, now=now)
    return task, tiled


def test_walkway_config_opts_in_cam0_and_carbon_do_not() -> None:
    p = _policy()
    cfg = p.tiled_detect_for("walkway", "walkway")
    assert cfg["enabled"] is True and float(cfg["min_seconds_between"]) > 0
    assert set(cfg["tiers"]) == {"baseline", "triggered"}
    assert p.tiled_detect_for("cam0", "cam0") == {}
    assert p.tiled_detect_for("carbon", "carbon") == {}


def test_walkway_request_gets_tiles_cam0_does_not() -> None:
    p, state = _policy(), RouterState()
    task, tiled = _task(p, state, "walkway", now=1000.0)
    assert tiled and task.request["detect_tiles"] is True
    cam_task, cam_tiled = _task(p, state, "cam0", now=1000.0)
    assert cam_task is not None and not cam_tiled
    assert "detect_tiles" not in cam_task.request


def test_tiles_are_spaced_by_min_seconds_between() -> None:
    p, state = _policy(), RouterState()
    gap = float(p.tiled_detect_for("walkway", "walkway")["min_seconds_between"])
    _, first = _task(p, state, "walkway", now=1000.0)
    assert first
    # Next dispatchable walkway frame inside the gap: untiled.
    state.camera("walkway").inflight.clear()
    t2, second = _task(p, state, "walkway", now=1000.0 + gap / 2)
    assert t2 is not None and not second and "detect_tiles" not in t2.request
    state.camera("walkway").inflight.clear()
    _, third = _task(p, state, "walkway", now=1000.0 + gap + 0.1)
    assert third


def test_tier_not_listed_gets_no_tiles() -> None:
    p, state = _policy(), RouterState()
    p.streams_cfg["walkway"] = dict(p.streams_cfg["walkway"])
    p.streams_cfg["walkway"]["tiled_detect"] = {"enabled": True, "tiers": ["triggered"], "min_seconds_between": 1}
    task, tiled = _task(p, state, "walkway", now=1000.0)  # nothing seen yet -> baseline tier
    assert task is not None and not tiled


def test_task_type_without_detect_gets_no_tiles() -> None:
    p, state = _policy(), RouterState()
    env = _env("walkway")
    for _ in range(20):
        d = p.decide(env, state, now=1000.0, image_path_exists=True)
        if d.should_dispatch:
            break
    frame = VisionFramePointerPayload.model_validate(env.payload)
    task = p.build_task_request(frame, env, d)
    task.task_type = "embed_image"
    assert not p.apply_tiled_detect(task, frame, d, state, now=1000.0)
    assert state.camera("walkway").last_tiled_detect_ts is None
