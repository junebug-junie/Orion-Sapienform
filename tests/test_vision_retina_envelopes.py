from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina"))

from app.envelopes import make_frame_pointer_envelope
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload


def test_frame_pointer_envelope_contract() -> None:
    payload = VisionFramePointerPayload(
        image_path="/mnt/telemetry/vision/frames/x.jpg",
        camera_id="retina-cam-01",
        stream_id="retina-stream-01",
        frame_ts=1710000000.0,
        width=640,
        height=480,
        format="jpg",
    )
    env = make_frame_pointer_envelope(
        payload,
        service_name="vision-retina",
        service_version="0.2.0",
    )
    assert isinstance(env, BaseEnvelope)
    assert env.kind == "vision.frame.pointer"
    assert env.schema_id == "orion.envelope"
    assert env.source == ServiceRef(name="vision-retina", version="0.2.0")
    validated = VisionFramePointerPayload.model_validate(env.payload)
    assert validated.camera_id == "retina-cam-01"
