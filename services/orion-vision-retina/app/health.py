from __future__ import annotations

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
            "last_error": metrics.last_error,
            "storage_dir": storage_dir,
        },
    )
    return BaseEnvelope(
        kind="system.health.v1",
        source=ServiceRef(name=service_name, version=service_version),
        payload=payload.model_dump(mode="json"),
    )
