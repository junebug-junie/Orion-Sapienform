from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina"))

from app.health import RetinaMetrics, make_system_health_envelope
from orion.schemas.telemetry.system_health import SystemHealthV1


def test_system_health_envelope_ok() -> None:
    metrics = RetinaMetrics(frames_published=3, fps_observed=0.9)
    env = make_system_health_envelope(
        service_name="vision-retina",
        service_version="0.2.0",
        camera_id="cam-01",
        stream_id="stream-01",
        source_type="folder",
        source_ok=True,
        metrics=metrics,
        fps_target=1.0,
        storage_dir="/tmp/frames",
    )
    assert env.kind == "system.health.v1"
    payload = SystemHealthV1.model_validate(env.payload)
    assert payload.status == "ok"
    assert payload.details["source_ok"] is True
    assert payload.details["fps_observed"] == 0.9


def test_system_health_envelope_degraded_on_failure() -> None:
    metrics = RetinaMetrics(
        frames_failed=2,
        last_error="source read returned no frame",
    )
    env = make_system_health_envelope(
        service_name="vision-retina",
        service_version="0.2.0",
        camera_id="cam-01",
        stream_id="stream-01",
        source_type="folder",
        source_ok=False,
        metrics=metrics,
        fps_target=1.0,
        storage_dir="/tmp/frames",
    )
    payload = SystemHealthV1.model_validate(env.payload)
    assert payload.status == "degraded"
    assert payload.details["last_error"] == "source read returned no frame"
