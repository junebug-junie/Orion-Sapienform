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
        return len(self.pending)

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
            if now - task.dispatched_at >= timeout_s:
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
