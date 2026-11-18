from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskRequestPayload, VisionTaskResultPayload

from .expectation import ExpectationCache
from .host_trigger import HOST_TRIGGER_SCORE_THRESHOLD
from .settings import Settings
from .state import PendingTask, RouterState

# Detections forwarded with one caption task (meta.source_objects), highest score first.
MAX_CAPTION_SOURCE_OBJECTS = 20

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


# Task types whose host pipeline runs GroundingDINO detect (config/
# vision_profiles.yaml task_routing); a tiles flag on anything else would
# spend the tiled_detect clock and do nothing.
TILED_DETECT_TASK_TYPES = frozenset({"retina_fast", "retina_dense", "detect_open_vocab"})


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
    # Why the triggered tier was chosen: "labels" (a trigger label was seen
    # recently), "expectation" (orion:vision:expect:<stream_id> is set, spec
    # idea 8) or "experiment" (orion:vision:experiment:<stream_id> = "on").
    # None on the baseline tier. Forwarded in task meta so the artifact
    # itself records whether prediction, detection or an experiment raised
    # the tier.
    triggered_by: str | None = None


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
    def __init__(
        self,
        *,
        settings: Settings,
        raw: dict[str, Any],
        expectation: ExpectationCache | None = None,
    ) -> None:
        self.settings = settings
        self.raw = raw
        self.expectation = expectation
        self.defaults: dict[str, Any] = dict(raw.get("defaults") or {})
        self.global_cfg: dict[str, Any] = dict(raw.get("global") or {})
        self.cameras_cfg: dict[str, Any] = dict(raw.get("cameras") or {})
        self.streams_cfg: dict[str, Any] = dict(raw.get("streams") or {})

    @classmethod
    def load(cls, settings: Settings, *, expectation: ExpectationCache | None = None) -> FrameDispatchPolicy:
        raw = load_policy_file(settings.ROUTER_POLICY_PATH)
        return cls(settings=settings, raw=raw, expectation=expectation)

    def default_trigger_labels(self) -> list[str]:
        normalized = _normalize_tiers(dict(self.defaults))
        triggered = dict(normalized.get("triggered") or {})
        return list(triggered.get("trigger_labels") or [])

    def trigger_labels_for(self, camera_id: str | None, stream_id: str | None) -> list[str]:
        """This stream's own trigger labels plus the defaults'. The host-reply
        path records only labels in this set, so a stream that adds its own
        trigger label (walkway's ``dog``) must widen it or that label can
        never raise the tier."""
        merged, _ = self.resolve_stream_policy(camera_id or "", stream_id)
        triggered = dict(_normalize_tiers(merged).get("triggered") or {})
        labels = list(self.default_trigger_labels())
        for lab in triggered.get("trigger_labels") or []:
            if lab not in labels:
                labels.append(lab)
        return labels

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
    ) -> tuple[str, dict[str, Any], str | None]:
        normalized = _normalize_tiers(merged_policy)
        baseline_cfg = dict(normalized.get("baseline") or {})
        triggered_cfg = dict(normalized.get("triggered") or {})

        trigger_labels = list(triggered_cfg.get("trigger_labels") or [])
        ttl_s = float(triggered_cfg.get("trigger_ttl_seconds") or 0)
        if self.expectation is not None and stream_id:
            self.expectation.note_stream(stream_id)

        # Experiment override (orion:vision:experiment:<stream_id> = "on", set
        # by orion-prediction's switchback runner with a TTL): forces the
        # triggered tier. Checked first so every on-block dispatch is
        # attributed to the experiment. Same opt-out rule as expectation: a
        # stream with no trigger labels has no triggered tier to force.
        if trigger_labels and self.expectation is not None and self.expectation.experiment_on(stream_id):
            tier_cfg = _shallow_merge(baseline_cfg, triggered_cfg)
            return "triggered", tier_cfg, "experiment"

        if (
            stream_id
            and trigger_labels
            and state.active_labels(stream_id, trigger_labels, ttl_s, now=now)
        ):
            tier_cfg = _shallow_merge(baseline_cfg, triggered_cfg)
            return "triggered", tier_cfg, "labels"

        # Expectation steering (spec idea 8): an open expectation window for
        # this stream raises it to the triggered tier even with nothing seen
        # yet. Only for streams that have trigger labels at all -- one with
        # an explicit empty list (porch_eye) has opted out of the tier.
        if trigger_labels and self.expectation is not None and self.expectation.is_open(stream_id):
            tier_cfg = _shallow_merge(baseline_cfg, triggered_cfg)
            return "triggered", tier_cfg, "expectation"

        return "baseline", baseline_cfg, None

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
        dispatch_tier, tier_cfg, triggered_by = self._tier_config(merged_policy, state, stream_id, now=now)

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
            triggered_by=triggered_by,
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
        if cam.identity_still_pending:
            return False
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
        if decision.triggered_by:
            meta["triggered_by"] = decision.triggered_by
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

    # ------------------------------------------------------------------
    # Tiled full-resolution detection (2026-10-15, walkway)
    # ------------------------------------------------------------------
    def tiled_detect_for(self, camera_id: str | None, stream_id: str | None) -> dict[str, Any]:
        """This stream's ``tiled_detect:`` block, or {} when absent/disabled.
        Opt-in per stream: cam0/carbon never ask for tiles."""
        merged, _ = self.resolve_stream_policy(camera_id or "", stream_id)
        cfg = dict(merged.get("tiled_detect") or {})
        return cfg if cfg.get("enabled", False) else {}

    def apply_tiled_detect(
        self,
        task: VisionTaskRequestPayload,
        frame: VisionFramePointerPayload,
        decision: FrameDispatchDecision,
        state: RouterState,
        *,
        now: float,
    ) -> bool:
        """Set ``request.detect_tiles`` on this primary task when the stream
        opted in, the tier is listed, and ``min_seconds_between`` has passed
        since the last tiled task for this camera. A tiled detect costs
        several GroundingDINO passes on the shared athena GPU, so it is
        spaced out rather than run on every frame. Geometry lives in the host
        profile (config/vision_profiles.yaml tile_* params)."""
        cfg = self.tiled_detect_for(frame.camera_id, frame.stream_id)
        if not cfg or task.task_type not in TILED_DETECT_TASK_TYPES:
            return False
        tiers = cfg.get("tiers") or ["baseline", "triggered"]
        if decision.dispatch_tier not in tiers:
            return False
        cam = state.camera(frame.camera_id or "unknown")
        gap = float(cfg.get("min_seconds_between", 15))
        if cam.last_tiled_detect_ts is not None and now - cam.last_tiled_detect_ts < gap:
            return False
        task.request["detect_tiles"] = True
        cam.last_tiled_detect_ts = now
        return True

    # ------------------------------------------------------------------
    # Walkway captions from the VLM lane (2026-10-15)
    # ------------------------------------------------------------------
    def caption_dispatch_for(self, camera_id: str | None, stream_id: str | None) -> dict[str, Any]:
        """This stream's ``caption_dispatch:`` block (stream-level, not per
        tier), or {} when absent/disabled. Opt-in per stream like
        identity_dispatch: cam0/carbon keep the host's own BLIP caption."""
        merged, _ = self.resolve_stream_policy(camera_id or "", stream_id)
        cfg = dict(merged.get("caption_dispatch") or {})
        return cfg if cfg.get("enabled", False) else {}

    @staticmethod
    def caption_gate_objects(result: VisionTaskResultPayload, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """The detections in a primary reply that let a caption be asked for:
        label in ``cfg.labels`` and score >= ``cfg.min_score``. Empty = the
        detector saw nobody, so no VLM call (a dark empty frame is where BLIP
        invented "a camera on a wooden bench in a garden"). Whether a detection
        is a stationary prop is decided downstream in orion-vision-window,
        which keeps the fixture history; these boxes travel with the caption
        task (meta.source_objects) so that gate can check them."""
        if not result.ok or result.artifact is None:
            return []
        labels = {str(x).strip().lower() for x in (cfg.get("labels") or []) if str(x).strip()}
        min_score = float(cfg.get("min_score", HOST_TRIGGER_SCORE_THRESHOLD))
        out: list[dict[str, Any]] = []
        for obj in result.artifact.outputs.objects or []:
            label = str(obj.label or "").strip().lower()
            if label not in labels or float(obj.score) < min_score:
                continue
            out.append({
                "label": label,
                "score": round(float(obj.score), 4),
                "box_xyxy": [float(v) for v in obj.box_xyxy],
            })
        out.sort(key=lambda o: o["score"], reverse=True)
        return out[:MAX_CAPTION_SOURCE_OBJECTS]

    @staticmethod
    def build_caption_task_request(
        task: PendingTask,
        *,
        stream_id: str | None,
        source_objects: list[dict[str, Any]],
        cfg: dict[str, Any],
    ) -> VisionTaskRequestPayload:
        """caption_frame for the VLM lane, same frame the detector just read."""
        request: dict[str, Any] = {}
        if task.image_path:
            request["image_path"] = task.image_path
        if task.percept_sha256:
            request["percept_sha256"] = task.percept_sha256
        prompt = str(cfg.get("prompt") or "").strip()
        if prompt:
            request["caption_prompt"] = prompt
        null_answers = [str(a) for a in (cfg.get("null_answers") or []) if str(a).strip()]
        if null_answers:
            request["caption_null_answers"] = null_answers
        if cfg.get("crop_to_detections") and source_objects:
            # The VLM reads a ~1 MP downscale of the frame; a person down the
            # street is a few pixels there. The host crops to these boxes
            # (orion-vision-host app/caption_crop.py).
            request["caption_crop_boxes"] = [list(o["box_xyxy"]) for o in source_objects]
            request["caption_crop_pad"] = float(cfg.get("crop_pad", 0.5))
        meta = {
            "camera_id": task.camera_id,
            "stream_id": stream_id,
            "frame_ts": task.frame_ts,
            "dispatch_tier": task.dispatch_tier,
            "caption_secondary_dispatch": True,
            "source_task_correlation_id": task.correlation_id,
            # Read by orion-vision-window's caption gate.
            "source_objects": source_objects,
        }
        return VisionTaskRequestPayload(task_type="caption_frame", request=request, meta=meta)
