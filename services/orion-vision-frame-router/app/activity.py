from __future__ import annotations

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionEdgeActivityPayload

from .state import RouterState


def handle_activity_envelope(env: BaseEnvelope, state: RouterState, *, now: float) -> None:
    payload = VisionEdgeActivityPayload.model_validate(env.payload)
    if payload.stream_id:
        state.record_activity(payload.stream_id, payload.labels, now=now)
