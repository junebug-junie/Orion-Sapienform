import sys
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app.activity import ActivityRateLimiter, labels_from_detections, publish_activity_if_allowed


def test_labels_from_detections_person_and_motion() -> None:
    detections = [
        {"kind": "yolo", "label": "person", "score": 0.9},
        {"kind": "motion", "label": "motion", "score": 1.0},
    ]
    assert labels_from_detections(detections) == ["person", "motion"]


def test_rate_limiter_blocks_duplicate_within_one_second() -> None:
    limiter = ActivityRateLimiter(min_interval_s=1.0)
    assert limiter.allow("cam0", "person", now=100.0) is True
    assert limiter.allow("cam0", "person", now=100.5) is False
    assert limiter.allow("cam0", "person", now=101.1) is True


def test_empty_detections_yields_no_labels() -> None:
    assert labels_from_detections([]) == []


class _MockBus:
    def __init__(self) -> None:
        self.enabled = True
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, env: BaseEnvelope) -> None:
        self.published.append((channel, env))


@pytest.mark.asyncio
async def test_publish_activity_if_allowed_publishes_person_detection() -> None:
    bus = _MockBus()
    settings = SimpleNamespace(
        SERVICE_NAME="orion-vision-edge",
        SERVICE_VERSION="0.2.0",
        CHANNEL_VISION_EDGE_ACTIVITY="orion:vision:edge:activity",
    )
    parent = BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-edge", version="0.2.0"),
        correlation_id=uuid4(),
        payload={},
    )
    limiter = ActivityRateLimiter(min_interval_s=0.0)
    detections = [{"kind": "yolo", "label": "person", "score": 0.9}]
    labels = labels_from_detections(detections)

    await publish_activity_if_allowed(
        bus,
        settings,
        stream_id="cam0",
        camera_id="rtsp://cam",
        labels=labels,
        max_score=0.9,
        frame_ts=time.time(),
        image_path="/tmp/f.jpg",
        artifact_id="art-1",
        limiter=limiter,
        parent_env=parent,
    )

    assert len(bus.published) == 1
    channel, env = bus.published[0]
    assert channel == "orion:vision:edge:activity"
    assert env.kind == "vision.edge.activity.v1"
    assert env.payload["labels"] == ["person"]
    assert env.payload["stream_id"] == "cam0"
