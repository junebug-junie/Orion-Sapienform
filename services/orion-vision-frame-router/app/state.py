from __future__ import annotations

from pydantic import BaseModel, Field


class CameraState(BaseModel):
    frames_seen: int = 0
    frames_dispatched: int = 0
    last_dispatch_ts: float | None = None
    # Separate from last_dispatch_ts -- identity_face is a secondary,
    # independently rate-limited dispatch alongside the primary tier's own
    # task (policy.py's decide_identity), not a replacement for it. Sharing
    # last_dispatch_ts would couple identity's cadence to whatever the
    # primary task's own min_seconds_between_tasks_per_camera happens to be.
    last_identity_dispatch_ts: float | None = None
    # A full-resolution still is being fetched for this camera's next face check
    # (dispatcher._identity_with_still); no second check starts until it is published.
    identity_still_pending: bool = False
    # Last walkway-style caption task sent to the VLM lane for this camera
    # (dispatcher._maybe_dispatch_caption); its own clock, like identity's.
    last_caption_dispatch_ts: float | None = None
    # The gate boxes of that last captioned frame: [{"label", "box_xyxy"}].
    last_caption_objects: list[dict] = Field(default_factory=list)
    # Last primary task this router marked detect_tiles (policy.apply_tiled_detect);
    # its own clock so tiled frames are spaced by tiled_detect.min_seconds_between.
    last_tiled_detect_ts: float | None = None
    inflight: set[str] = Field(default_factory=set)
    last_skip_reason: str | None = None


class PendingTask(BaseModel):
    correlation_id: str
    camera_id: str
    stream_id: str | None = None
    frame_ts: float | None
    image_path: str
    task_type: str
    dispatched_at: float
    reply_to: str
    # Read by the vision organ recorder (app/grammar_emit.py) when the reply lands:
    # only a primary task's reply counts toward detection/caption yield.
    is_primary: bool = True
    want_caption: bool = False
    # "host" = the athena vision host (counts toward max_inflight_total);
    # "caption" = the separate Qwen VLM lane, capped by its own
    # caption_dispatch.max_inflight, never by the athena host's cap.
    lane: str = "host"
    dispatch_tier: str | None = None
    percept_sha256: str | None = None
    # Per-task timeout (caption lane); None = TASK_TIMEOUT_SECONDS.
    timeout_s: float | None = None


class StreamTriggerState(BaseModel):
    label_ts: dict[str, float] = Field(default_factory=dict)


class RouterState:
    def __init__(self) -> None:
        self.cameras: dict[str, CameraState] = {}
        self.pending: dict[str, PendingTask] = {}
        self.stream_triggers: dict[str, StreamTriggerState] = {}

    def camera(self, camera_id: str) -> CameraState:
        if camera_id not in self.cameras:
            self.cameras[camera_id] = CameraState()
        return self.cameras[camera_id]

    def inflight_total(self) -> int:
        """Tasks in flight on the athena vision host -- what
        ``max_inflight_total`` caps. Caption-lane tasks run on a different
        GPU and are capped separately (``inflight_lane("caption")``)."""
        return sum(1 for t in self.pending.values() if t.lane == "host")

    def inflight_lane(self, lane: str) -> int:
        return sum(1 for t in self.pending.values() if t.lane == lane)

    def mark_seen(self, camera_id: str) -> CameraState:
        cam = self.camera(camera_id)
        cam.frames_seen += 1
        return cam

    def mark_dispatched(
        self,
        *,
        correlation_id: str,
        camera_id: str,
        image_path: str,
        task_type: str,
        reply_to: str,
        now: float,
        frame_ts: float | None,
        stream_id: str | None = None,
        is_primary: bool = True,
        want_caption: bool = False,
        lane: str = "host",
        dispatch_tier: str | None = None,
        percept_sha256: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """``is_primary=False`` for the identity_face secondary dispatch
        (dispatcher.py): review finding, 2026-08-26, independently
        surfaced by three separate review passes -- an unconditional
        ``cam.inflight.add()`` here meant identity's own corr_id consumed
        the SAME per-camera slot ``max_inflight_per_camera`` gates for the
        PRIMARY tier's own decide() calls (live cam0 config:
        max_inflight_per_camera=1), so once identity fired, cam0 could not
        dispatch another retina_fast until identity's own reply/timeout
        cleared -- freezing primary detection for up to
        TASK_TIMEOUT_SECONDS, exactly while a person is present. The same
        unconditional stamp also re-paced ``last_dispatch_ts`` (the
        PRIMARY tier's own ``min_seconds_between_tasks_per_camera`` clock),
        contradicting ``last_identity_dispatch_ts``'s whole reason for
        being a separate field. The task is still tracked in
        ``self.pending`` either way (global inflight, corr_id lookup,
        timeout sweep all still apply identically) -- only the PRIMARY-
        tier-specific per-camera counters are skipped for a secondary
        dispatch.
        """
        cam = self.camera(camera_id)
        if is_primary:
            cam.frames_dispatched += 1
            cam.last_dispatch_ts = now
            cam.inflight.add(correlation_id)
        self.pending[correlation_id] = PendingTask(
            correlation_id=correlation_id,
            camera_id=camera_id,
            stream_id=stream_id,
            frame_ts=frame_ts,
            image_path=image_path,
            task_type=task_type,
            dispatched_at=now,
            reply_to=reply_to,
            is_primary=is_primary,
            want_caption=want_caption,
            lane=lane,
            dispatch_tier=dispatch_tier,
            percept_sha256=percept_sha256,
            timeout_s=timeout_s,
        )

    def clear_pending(self, correlation_id: str, *, now: float) -> PendingTask | None:
        task = self.pending.pop(correlation_id, None)
        if not task:
            return None
        cam = self.camera(task.camera_id)
        cam.inflight.discard(correlation_id)
        return task

    def expired_correlation_ids(self, *, now: float, timeout_s: float) -> list[str]:
        out: list[str] = []
        for cid, task in list(self.pending.items()):
            limit = task.timeout_s if task.timeout_s is not None else timeout_s
            if now - task.dispatched_at >= limit:
                out.append(cid)
        return out

    def record_activity(self, stream_id: str, labels: list[str], *, now: float) -> None:
        st = self.stream_triggers.setdefault(stream_id, StreamTriggerState())
        for label in labels:
            st.label_ts[label.lower()] = now

    def active_labels(
        self, stream_id: str, trigger_labels: list[str], ttl_s: float, *, now: float
    ) -> list[str]:
        st = self.stream_triggers.get(stream_id)
        if not st:
            return []
        out: list[str] = []
        for label in trigger_labels:
            ts = st.label_ts.get(label.lower())
            if ts is not None and (now - ts) <= ttl_s:
                out.append(label.lower())
        return out
