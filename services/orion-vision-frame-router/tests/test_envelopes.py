from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskRequestPayload

from app.envelopes import make_host_task_envelope


def test_host_task_envelope_contract() -> None:
    corr = uuid4()
    frame = VisionFramePointerPayload(image_path="/mnt/x.jpg", camera_id="cam1", frame_ts=1.0)
    frame_env = BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-retina", version="0.2.0"),
        correlation_id=corr,
        payload=frame.model_dump(mode="json"),
    )
    task = VisionTaskRequestPayload(
        task_type="retina_fast",
        request={"image_path": "/mnt/x.jpg"},
        meta={"camera_id": "cam1"},
    )
    reply_to = f"orion:vision:reply:{corr}"
    out = make_host_task_envelope(
        frame_env=frame_env,
        frame=frame,
        task=task,
        service_name="vision-frame-router",
        service_version="0.1.0",
        reply_to=reply_to,
    )
    assert out.kind == "vision.task.request"
    assert out.reply_to == reply_to
    assert out.correlation_id == corr
    assert len(out.causality_chain) == 1
    assert out.causality_chain[0].kind == "vision.frame.pointer"
    validated = VisionTaskRequestPayload.model_validate(out.payload)
    assert validated.task_type == "retina_fast"
