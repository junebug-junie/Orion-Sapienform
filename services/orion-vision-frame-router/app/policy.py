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

_LEGACY_TIER_KEYS = frozenset(
    {
        "task_type",
        "every_n_frames",
        "min_seconds_between_tasks_per_camera",
        "max_inflight_per_camera",
        "request",
    }
)


class FrameDispatchDecision(BaseModel):
    should_dispatch: bool
    task_type: str | None = None
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    policy_name: str = "defaults"
    dispatch_tier: str = "baseline"
    reason: str = ""
    # This tier's `identity_dispatch:` config block, verbatim (may be empty).
    # Carried on the decision rather than re-resolved by the caller so
    # decide_identity() can be a small, pure function operating on already-
    # computed data instead of re-walking policy internals.
    identity_dispatch_cfg: dict[str, Any] = Field(default_factory=dict)


def load_policy_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"policy file must be a mapping: {path}")
    return data


def _shallow_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _normalize_tiers(policy: dict[str, Any]) -> dict[str, Any]:
    result = dict(policy)
    baseline = dict(result.get("baseline") or {})
    for key in _LEGACY_TIER_KEYS:
        if key in result:
            baseline[key] = result.pop(key)
    if baseline or "baseline" in policy or "triggered" in policy:
        result["baseline"] = baseline
    if "triggered" not in result:
        result["triggered"] = dict(policy.get("triggered") or {})
    return result


class FrameDispatchPolicy:
    def __init__(self, *, settings: Settings, raw: dict[str, Any]) -> None:
        self.settings = settings
        self.raw = raw
        self.defaults: dict[str, Any] = dict(raw.get("defaults") or {})
        self.global_cfg: dict[str, Any] = dict(raw.get("global") or {})
        self.cameras_cfg: dict[str, Any] = dict(raw.get("cameras") or {})
        self.streams_cfg: dict[str, Any] = dict(raw.get("streams") or {})

    @classmethod
    def load(cls, settings: Settings) -> FrameDispatchPolicy:
        raw = load_policy_file(settings.ROUTER_POLICY_PATH)
        return cls(settings=settings, raw=raw)

    def default_trigger_labels(self) -> list[str]:
        normalized = _normalize_tiers(dict(self.defaults))
        triggered = dict(normalized.get("triggered") or {})
        return list(triggered.get("trigger_labels") or [])

    def resolve_camera_policy(self, camera_id: str) -> tuple[dict[str, Any], str]:
        merged, name = self.resolve_stream_policy(camera_id, None)
        return merged, name

    def resolve_stream_policy(
        self, camera_id: str, stream_id: str | None
    ) -> tuple[dict[str, Any], str]:
        merged = dict(self.defaults)
        policy_name = "defaults"

        if stream_id and stream_id in self.streams_cfg:
            merged = _shallow_merge(merged, dict(self.streams_cfg[stream_id]))
            policy_name = stream_id

        if camera_id in self.cameras_cfg:
            merged = _shallow_merge(merged, dict(self.cameras_cfg[camera_id]))
            policy_name = camera_id

        return merged, policy_name

    def _tier_config(
        self, merged_policy: dict[str, Any], state: RouterState, stream_id: str | None, *, now: float
    ) -> tuple[str, dict[str, Any]]:
        normalized = _normalize_tiers(merged_policy)
        baseline_cfg = dict(normalized.get("baseline") or {})
        triggered_cfg = dict(normalized.get("triggered") or {})

        trigger_labels = list(triggered_cfg.get("trigger_labels") or [])
        ttl_s = float(triggered_cfg.get("trigger_ttl_seconds") or 0)

        if (
            stream_id
            and trigger_labels
            and state.active_labels(stream_id, trigger_labels, ttl_s, now=now)
        ):
            tier_cfg = _shallow_merge(baseline_cfg, triggered_cfg)
            return "triggered", tier_cfg

        return "baseline", baseline_cfg

    def require_image_path_exists(self, camera_id: str) -> bool:
        cam_policy, _ = self.resolve_camera_policy(camera_id)
        return bool(
            cam_policy.get(
                "require_image_path_exists",
                self.global_cfg.get("require_image_path_exists", self.settings.REQUIRE_IMAGE_PATH_EXISTS),
            )
        )

    def drop_when_busy(self) -> bool:
        return bool(self.global_cfg.get("drop_when_busy", True))

    def decide(
        self,
        env: BaseEnvelope,
        state: RouterState,
        *,
        now: float,
        image_path_exists: bool | None = None,
    ) -> FrameDispatchDecision:
        if not self.settings.ROUTER_ENABLED:
            return FrameDispatchDecision(should_dispatch=False, policy_name="env", reason="router_disabled")

        frame = VisionFramePointerPayload.model_validate(env.payload)
        camera_id = frame.camera_id or "unknown"
        stream_id = frame.stream_id
        cam_state = state.mark_seen(camera_id)
        merged_policy, policy_name = self.resolve_stream_policy(camera_id, stream_id)
        dispatch_tier, tier_cfg = self._tier_config(merged_policy, state, stream_id, now=now)

        if not merged_policy.get("enabled", True):
            cam_state.last_skip_reason = "camera_disabled"
            return FrameDispatchDecision(
                should_dispatch=False,
                policy_name=policy_name,
                dispatch_tier=dispatch_tier,
                reason="camera_disabled",
            )

        image_path = (frame.image_path or "").strip()
        sha256 = (frame.sha256 or "").strip()
        require_exists = self.require_image_path_exists(camera_id)

        # A content-addressed frame is addressed, full stop. It lives in
        # orion-percept-store, not on a filesystem this router can stat, so the
        # path checks below do not apply to it -- and applying them would reject
        # every frame from every node that does not share athena's disk, which
        # is the entire reason sha256 exists.
        #
        # There is deliberately no "neither address" branch here. It would be
        # unreachable: `VisionFramePointerPayload` rejects an addressless
        # pointer at validation, and this method re-validates above (as does
        # the dispatcher, which reports it as `invalid_frame_payload` and skips
        # the frame). A first draft of this change did add such a branch; it
        # could never execute, which is why it is gone.
        if sha256 and not image_path:
            pass
        elif require_exists:
            visible = image_path_exists if image_path_exists is not None else os.path.isfile(image_path)
            if not visible:
                cam_state.last_skip_reason = "image_path_not_visible"
                return FrameDispatchDecision(
                    should_dispatch=False,
                    policy_name=policy_name,
                    dispatch_tier=dispatch_tier,
                    reason="image_path_not_visible",
                )

        every_n = int(tier_cfg.get("every_n_frames", self.settings.DEFAULT_EVERY_N_FRAMES))
        if every_n > 1 and (cam_state.frames_seen % every_n) != 0:
            cam_state.last_skip_reason = "frame_sampled_out"
            return FrameDispatchDecision(
                should_dispatch=False,
                policy_name=policy_name,
                dispatch_tier=dispatch_tier,
                reason="frame_sampled_out",
            )

        min_s = float(
            tier_cfg.get("min_seconds_between_tasks_per_camera", self.settings.DEFAULT_MIN_SECONDS_PER_CAMERA)
        )
        if cam_state.last_dispatch_ts is not None and (now - cam_state.last_dispatch_ts) < min_s:
            cam_state.last_skip_reason = "camera_rate_limited"
            return FrameDispatchDecision(
                should_dispatch=False,
                policy_name=policy_name,
                dispatch_tier=dispatch_tier,
                reason="camera_rate_limited",
            )

        max_cam = int(tier_cfg.get("max_inflight_per_camera", self.settings.MAX_INFLIGHT_PER_CAMERA))
        if len(cam_state.inflight) >= max_cam:
            cam_state.last_skip_reason = "camera_inflight_limit"
            return FrameDispatchDecision(
                should_dispatch=False,
                policy_name=policy_name,
                dispatch_tier=dispatch_tier,
                reason="camera_inflight_limit",
            )

        max_total = int(self.global_cfg.get("max_inflight_total", self.settings.MAX_INFLIGHT_TOTAL))
        if state.inflight_total() >= max_total:
            reason = "global_inflight_limit" if self.drop_when_busy() else "global_inflight_backpressure"
            cam_state.last_skip_reason = reason
            return FrameDispatchDecision(
                should_dispatch=False,
                policy_name=policy_name,
                dispatch_tier=dispatch_tier,
                reason=reason,
            )

        task_type = str(tier_cfg.get("task_type", self.settings.DEFAULT_TASK_TYPE))
        if task_type not in SUPPORTED_TASK_TYPES:
            cam_state.last_skip_reason = "unsupported_task_type"
            return FrameDispatchDecision(
                should_dispatch=False,
                policy_name=policy_name,
                dispatch_tier=dispatch_tier,
                reason="unsupported_task_type",
            )

        request_overrides = dict(tier_cfg.get("request") or {})
        return FrameDispatchDecision(
            should_dispatch=True,
            task_type=task_type,
            request_overrides=request_overrides,
            policy_name=policy_name,
            dispatch_tier=dispatch_tier,
            reason="dispatch",
            identity_dispatch_cfg=dict(tier_cfg.get("identity_dispatch") or {}),
        )

    def decide_identity(
        self,
        decision: FrameDispatchDecision,
        *,
        camera_id: str,
        state: RouterState,
        now: float,
    ) -> bool:
        """Whether to ALSO dispatch identity_face for this frame, alongside
        the tier's own primary task -- a second, independent decision, not
        a mode of decide() above. decide()'s model is one task_type per
        tier; identity augments the primary detection rather than replacing
        it, so it needed its own decision rather than overloading task_type
        (which would mean a triggered frame gets identity_face INSTEAD of
        retina_fast, losing the object detection that still matters).

        Deliberately opt-in per stream (config's `triggered.identity_dispatch.
        enabled: true`), not a global default: every stream that never
        overrides `triggered` (e.g. porch_eye's `trigger_labels: []`)
        already never reaches the "triggered" tier at all and is therefore
        already safe regardless of a permissive default, but a bare
        `defaults.triggered.identity_dispatch.enabled: true` would still
        reach every stream that DOES use plain trigger_labels defaults --
        including carbon, which the design doc (docs/superpowers/specs/
        2026-08-21-...-design.md §9B) explicitly says must not inherit
        cam0's policy. Opt-in per stream keeps that a deliberate choice,
        not an accident of shallow-merge inheritance.

        Only ever considered on the SAME "triggered" condition the tier
        already computed (e.g. person seen recently) -- piggybacks on
        existing trigger state rather than running a second detector just
        to decide whether to look for a face. Rate-limited independently
        of the primary task's own min_seconds_between_tasks_per_camera
        (CameraState.last_identity_dispatch_ts, a separate field) since
        identity_face is real GPU cost that should not fire on every single
        triggered frame. Gated on the GLOBAL inflight cap only, not
        max_inflight_per_camera -- that cap is usually 1 and the primary
        task for THIS SAME frame already occupies it by the time this is
        called, which would make the feature permanently inert on exactly
        the config it targets.
        """
        if decision.dispatch_tier != "triggered":
            return False
        cfg = decision.identity_dispatch_cfg
        if not cfg.get("enabled", False):
            return False
        min_s = float(cfg.get("min_seconds_between_dispatch", self.settings.DEFAULT_IDENTITY_MIN_SECONDS))
        cam = state.camera(camera_id)
        if cam.last_identity_dispatch_ts is not None and (now - cam.last_identity_dispatch_ts) < min_s:
            return False
        max_total = int(self.global_cfg.get("max_inflight_total", self.settings.MAX_INFLIGHT_TOTAL))
        if state.inflight_total() >= max_total:
            return False
        return True

    def build_task_request(
        self,
        frame: VisionFramePointerPayload,
        env: BaseEnvelope,
        decision: FrameDispatchDecision,
    ) -> VisionTaskRequestPayload:
        # Forward whichever address the frame actually carries. A frame from a
        # node with no shared filesystem has only a sha256, and the host
        # resolves it from orion-percept-store; a local frame keeps its path,
        # which stays the cheap route (no HTTP hop, no copy).
        base_request = self._frame_address(frame)
        base_request.update(decision.request_overrides or {})
        meta = {
            "camera_id": frame.camera_id,
            "stream_id": frame.stream_id,
            "frame_ts": frame.frame_ts,
            "source_frame_envelope_id": str(env.id),
            "source_frame_correlation_id": str(env.correlation_id),
            "router_policy": decision.policy_name,
            "dispatch_tier": decision.dispatch_tier,
        }
        return VisionTaskRequestPayload(
            task_type=decision.task_type or "retina_fast",
            request=base_request,
            meta=meta,
        )

    @staticmethod
    def _frame_address(frame: VisionFramePointerPayload) -> dict[str, Any]:
        # image_path / percept_sha256 -- shared by build_task_request above
        # and build_identity_task_request below, factored out rather than
        # duplicated. Note for the structural test in test_content_
        # addressed_frames.py (greps from "def build_task_request" onward):
        # this must stay defined AFTER build_task_request in this file, or
        # the "percept_sha256"/"image_path" literals it checks for fall
        # outside that slice again.
        base_request: dict[str, Any] = {}
        if frame.image_path:
            base_request["image_path"] = frame.image_path
        if frame.sha256:
            base_request["percept_sha256"] = frame.sha256
        return base_request

    def build_identity_task_request(
        self,
        frame: VisionFramePointerPayload,
        env: BaseEnvelope,
        decision: FrameDispatchDecision,
    ) -> VisionTaskRequestPayload:
        """The secondary identity_face task alongside `decision`'s own
        primary task. Same frame address, own task_type, no
        request_overrides -- those are the PRIMARY tier's own request
        knobs (want_caption, want_embeddings, ...), not identity_face
        params (identity's thresholds/max_candidates live in config/
        vision_profiles.yaml's profile block, not per-request)."""
        base_request = self._frame_address(frame)
        meta = {
            "camera_id": frame.camera_id,
            "stream_id": frame.stream_id,
            "frame_ts": frame.frame_ts,
            "source_frame_envelope_id": str(env.id),
            "source_frame_correlation_id": str(env.correlation_id),
            "router_policy": decision.policy_name,
            "dispatch_tier": decision.dispatch_tier,
            "identity_secondary_dispatch": True,
        }
        return VisionTaskRequestPayload(
            task_type="identity_face",
            request=base_request,
            meta=meta,
        )
