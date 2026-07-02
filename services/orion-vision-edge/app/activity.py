from __future__ import annotations

import time
import uuid
from typing import Any

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionEdgeActivityPayload

VISION_EDGE_ACTIVITY_KIND = "vision.edge.activity.v1"
TRIGGER_LABELS = frozenset({"person", "motion"})


def labels_from_detections(detections: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for det in detections:
        kind = str(det.get("kind") or "")
        label = str(det.get("label") or kind or "").strip().lower()
        if not label:
            continue
        if kind == "motion" or label == "motion":
            label = "motion"
        if label not in TRIGGER_LABELS:
            continue
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


class ActivityRateLimiter:
    def __init__(self, min_interval_s: float = 1.0) -> None:
        self.min_interval_s = min_interval_s
        self._last: dict[tuple[str, str], float] = {}

    def allow(self, stream_id: str, label: str, *, now: float) -> bool:
        key = (stream_id, label)
        prev = self._last.get(key)
        if prev is not None and (now - prev) < self.min_interval_s:
            return False
        self._last[key] = now
        return True


def build_activity_payload(
    *,
    stream_id: str,
    camera_id: str | None,
    labels: list[str],
    max_score: float,
    frame_ts: float | None,
    image_path: str | None,
    artifact_id: str | None,
) -> VisionEdgeActivityPayload:
    return VisionEdgeActivityPayload(
        stream_id=stream_id,
        camera_id=camera_id,
        labels=labels,
        max_score=max_score,
        frame_ts=frame_ts,
        image_path=image_path,
        artifact_id=artifact_id,
    )


async def publish_activity_if_allowed(
    bus,
    settings,
    *,
    stream_id: str,
    camera_id: str | None,
    labels: list[str],
    max_score: float,
    frame_ts: float | None,
    image_path: str | None,
    artifact_id: str | None,
    limiter: ActivityRateLimiter,
    parent_env: BaseEnvelope,
) -> None:
    if not labels or not bus.enabled:
        return
    now = time.time()
    allowed = [lb for lb in labels if limiter.allow(stream_id, lb, now=now)]
    if not allowed:
        return
    payload = build_activity_payload(
        stream_id=stream_id,
        camera_id=camera_id,
        labels=allowed,
        max_score=max_score,
        frame_ts=frame_ts,
        image_path=image_path,
        artifact_id=artifact_id or str(uuid.uuid4()),
    )
    out = parent_env.derive_child(
        kind=VISION_EDGE_ACTIVITY_KIND,
        source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
        payload=payload.model_dump(mode="json"),
    )
    await bus.publish(settings.CHANNEL_VISION_EDGE_ACTIVITY, out)
