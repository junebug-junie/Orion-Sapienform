from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.system_health import SystemHealthV1


BOOT_ID = str(uuid.uuid4())


@dataclass
class RetinaMetrics:
    frames_published: int = 0
    frames_failed: int = 0
    last_frame_ts: float | None = None
    last_error: str | None = None
    fps_observed: float = 0.0
    # Times an RTSP reader was abandoned and replaced for delivering no frame
    # (RtspFrameSource stall watchdog). Always 0 for other sources.
    source_stall_restarts: int = 0
    # RTSP session hygiene (RtspFrameSource quiet period + self-heal).
    # Always 0/None for other sources.
    source_quiet_wait_remaining_sec: float = 0.0
    source_last_session_end_ts: float | None = None
    source_corrupt_frame_rate: float | None = None
    source_corruption_checks: int = 0
    source_corrupt_frames: int = 0
    source_corruption_heals: int = 0
    source_corruption_heal_suppressed: bool = False


def last_frame_age_sec(metrics: RetinaMetrics, now: float | None = None) -> float | None:
    """Seconds since the newest published frame was captured, or None before
    the first. The number that grows when a camera goes quiet."""
    if metrics.last_frame_ts is None:
        return None
    return round(max(0.0, (now if now is not None else time.time()) - metrics.last_frame_ts), 3)


def make_system_health_envelope(
    *,
    service_name: str,
    service_version: str,
    camera_id: str,
    stream_id: str,
    source_type: str,
    source_ok: bool,
    metrics: RetinaMetrics,
    fps_target: float,
    storage_dir: str,
    heartbeat_interval_sec: float,
) -> BaseEnvelope:
    # Threaded from the caller's real loop period rather than left at the schema
    # default: orion-equilibrium-service computes grace = interval * 3.0 and marks
    # the service "down" past it, so a hardcoded 10.0 silently mislabels any
    # deployment that overrides HEALTH_INTERVAL_SECONDS.
    payload = SystemHealthV1(
        heartbeat_interval_sec=heartbeat_interval_sec,
        service=service_name,
        version=service_version,
        boot_id=BOOT_ID,
        last_seen_ts=datetime.now(timezone.utc),
        status="ok" if source_ok else "degraded",
        details={
            "camera_id": camera_id,
            "stream_id": stream_id,
            "source_type": source_type,
            "source_ok": source_ok,
            "frames_published": metrics.frames_published,
            "frames_failed": metrics.frames_failed,
            "fps_target": fps_target,
            "fps_observed": metrics.fps_observed,
            "last_frame_ts": metrics.last_frame_ts,
            "last_frame_age_sec": last_frame_age_sec(metrics),
            "source_stall_restarts": metrics.source_stall_restarts,
            "source_quiet_wait_remaining_sec": metrics.source_quiet_wait_remaining_sec,
            "source_last_session_end_ts": metrics.source_last_session_end_ts,
            "source_corrupt_frame_rate": metrics.source_corrupt_frame_rate,
            "source_corruption_checks": metrics.source_corruption_checks,
            "source_corrupt_frames": metrics.source_corrupt_frames,
            "source_corruption_heals": metrics.source_corruption_heals,
            "source_corruption_heal_suppressed": metrics.source_corruption_heal_suppressed,
            "last_error": metrics.last_error,
            "storage_dir": storage_dir,
        },
    )
    return BaseEnvelope(
        kind="system.health.v1",
        source=ServiceRef(name=service_name, version=service_version),
        payload=payload.model_dump(mode="json"),
    )
