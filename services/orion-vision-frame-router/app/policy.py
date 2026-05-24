from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskRequestPayload

from .settings import Settings
from .state import RouterState

SUPPORTED_TASK_TYPES = {"retina_fast", "embed_image", "detect_open_vocab", "caption_frame"}


class FrameDispatchDecision(BaseModel):
    should_dispatch: bool
    task_type: str | None = None
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    policy_name: str = "defaults"
    reason: str = ""


def load_policy_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"policy file must be a mapping: {path}")
    return data


class FrameDispatchPolicy:
    def __init__(self, *, settings: Settings, raw: dict[str, Any]) -> None:
        self.settings = settings
        self.raw = raw
        self.defaults: dict[str, Any] = dict(raw.get("defaults") or {})
        self.global_cfg: dict[str, Any] = dict(raw.get("global") or {})
        self.cameras_cfg: dict[str, Any] = dict(raw.get("cameras") or {})

    @classmethod
    def load(cls, settings: Settings) -> FrameDispatchPolicy:
        raw = load_policy_file(settings.ROUTER_POLICY_PATH)
        return cls(settings=settings, raw=raw)

    def resolve_camera_policy(self, camera_id: str) -> tuple[dict[str, Any], str]:
        if camera_id in self.cameras_cfg:
            override = dict(self.cameras_cfg[camera_id])
            merged = {**self.defaults, **override}
            return merged, camera_id
        return dict(self.defaults), "defaults"

    def decide(self, env: BaseEnvelope, state: RouterState, *, now: float) -> FrameDispatchDecision:
        if not self.settings.ROUTER_ENABLED:
            return FrameDispatchDecision(should_dispatch=False, policy_name="env", reason="router_disabled")

        frame = VisionFramePointerPayload.model_validate(env.payload)
        camera_id = frame.camera_id or "unknown"
        cam_state = state.mark_seen(camera_id)
        cam_policy, policy_name = self.resolve_camera_policy(camera_id)

        if not cam_policy.get("enabled", True):
            cam_state.last_skip_reason = "camera_disabled"
            return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="camera_disabled")

        image_path = (frame.image_path or "").strip()
        require_exists = bool(
            cam_policy.get(
                "require_image_path_exists",
                self.global_cfg.get("require_image_path_exists", self.settings.REQUIRE_IMAGE_PATH_EXISTS),
            )
        )
        if not image_path:
            cam_state.last_skip_reason = "missing_image_path"
            return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="missing_image_path")
        if require_exists and not os.path.isfile(image_path):
            cam_state.last_skip_reason = "image_path_not_visible"
            return FrameDispatchDecision(
                should_dispatch=False, policy_name=policy_name, reason="image_path_not_visible"
            )

        every_n = int(cam_policy.get("every_n_frames", self.settings.DEFAULT_EVERY_N_FRAMES))
        if every_n > 1 and (cam_state.frames_seen % every_n) != 0:
            cam_state.last_skip_reason = "frame_sampled_out"
            return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="frame_sampled_out")

        min_s = float(
            cam_policy.get("min_seconds_between_tasks_per_camera", self.settings.DEFAULT_MIN_SECONDS_PER_CAMERA)
        )
        if cam_state.last_dispatch_ts is not None and (now - cam_state.last_dispatch_ts) < min_s:
            cam_state.last_skip_reason = "camera_rate_limited"
            return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="camera_rate_limited")

        max_cam = int(cam_policy.get("max_inflight_per_camera", self.settings.MAX_INFLIGHT_PER_CAMERA))
        if len(cam_state.inflight) >= max_cam:
            cam_state.last_skip_reason = "camera_inflight_limit"
            return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="camera_inflight_limit")

        max_total = int(self.global_cfg.get("max_inflight_total", self.settings.MAX_INFLIGHT_TOTAL))
        if state.inflight_total() >= max_total:
            cam_state.last_skip_reason = "global_inflight_limit"
            return FrameDispatchDecision(should_dispatch=False, policy_name=policy_name, reason="global_inflight_limit")

        task_type = str(cam_policy.get("task_type", self.settings.DEFAULT_TASK_TYPE))
        if task_type not in SUPPORTED_TASK_TYPES:
            cam_state.last_skip_reason = "unsupported_task_type"
            return FrameDispatchDecision(
                should_dispatch=False, policy_name=policy_name, reason="unsupported_task_type"
            )

        request_overrides = dict(cam_policy.get("request") or {})
        return FrameDispatchDecision(
            should_dispatch=True,
            task_type=task_type,
            request_overrides=request_overrides,
            policy_name=policy_name,
            reason="dispatch",
        )

    def build_task_request(
        self,
        frame: VisionFramePointerPayload,
        env: BaseEnvelope,
        decision: FrameDispatchDecision,
    ) -> VisionTaskRequestPayload:
        base_request: dict[str, Any] = {"image_path": frame.image_path}
        base_request.update(decision.request_overrides or {})
        meta = {
            "camera_id": frame.camera_id,
            "stream_id": frame.stream_id,
            "frame_ts": frame.frame_ts,
            "source_frame_envelope_id": str(env.id),
            "source_frame_correlation_id": str(env.correlation_id),
            "router_policy": decision.policy_name,
        }
        return VisionTaskRequestPayload(
            task_type=decision.task_type or "retina_fast",
            request=base_request,
            meta=meta,
        )
