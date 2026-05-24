from __future__ import annotations

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskRequestPayload


def make_host_task_envelope(
    *,
    frame_env: BaseEnvelope,
    frame: VisionFramePointerPayload,
    task: VisionTaskRequestPayload,
    service_name: str,
    service_version: str,
    reply_to: str,
) -> BaseEnvelope:
    _ = frame  # lineage anchor; payload lives in task
    return frame_env.derive_child(
        kind="vision.task.request",
        source=ServiceRef(name=service_name, version=service_version),
        payload=task,
        reply_to=reply_to,
    )
