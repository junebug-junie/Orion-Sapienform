from __future__ import annotations

import uuid

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload


def make_frame_pointer_envelope(
    payload: VisionFramePointerPayload,
    *,
    service_name: str,
    service_version: str,
    correlation_id: str | None = None,
) -> BaseEnvelope:
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name=service_name, version=service_version),
        correlation_id=uuid.UUID(correlation_id) if correlation_id else uuid.uuid4(),
        payload=payload.model_dump(mode="json"),
    )
