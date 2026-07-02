from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionEdgeActivityPayload

from app.activity import handle_activity_envelope
from app.state import RouterState


def _activity_env(stream_id: str = "cam0", labels: list[str] | None = None) -> BaseEnvelope:
    payload = VisionEdgeActivityPayload(
        stream_id=stream_id,
        labels=labels or ["person"],
        max_score=0.9,
    )
    return BaseEnvelope(
        kind="vision.edge.activity.v1",
        source=ServiceRef(name="vision-edge", version="0.2.0"),
        correlation_id=uuid4(),
        payload=payload.model_dump(mode="json"),
    )


def test_handle_activity_envelope_records_labels_for_ttl() -> None:
    state = RouterState()
    handle_activity_envelope(_activity_env(labels=["person", "motion"]), state, now=100.0)
    active = state.active_labels("cam0", ["person", "motion"], ttl_s=8.0, now=100.5)
    assert active == ["person", "motion"]


def test_handle_activity_envelope_expires_after_ttl() -> None:
    state = RouterState()
    handle_activity_envelope(_activity_env(labels=["person"]), state, now=100.0)
    assert state.active_labels("cam0", ["person"], ttl_s=8.0, now=107.0) == ["person"]
    assert state.active_labels("cam0", ["person"], ttl_s=8.0, now=108.1) == []
