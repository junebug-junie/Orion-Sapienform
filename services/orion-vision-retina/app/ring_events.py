"""Ring pushes -> recorded events (spec 2026-10-20 street continuity, sections 1 and 6).

A Ring push is a fact of the doorbell, not of a frame, so the retina writes it
itself, on two rails:

* ``orion:vision:events`` (``vision.event.bundle``, ``VisionEventPayload``) ->
  orion-vision-scribe -> ``orion:vision:events:sql-write`` -> orion-sql-writer
  -> one ``vision_events`` row. The same path orion-vision-window uses for
  ``camera_caption`` rows. event_type ``doorbell_ring`` (salience 1.0) or
  ``doorbell_motion`` (salience 0.2), stream_id ``doorbell``. The envelope's
  correlation_id is the push's correlation id, which the sql-writer stores in
  ``vision_events.correlation_id`` and which the retina's log line names.
* ``orion:vision:street:events`` (``StreetEventV1``), for a ring only:
  ``doorbell_ring``, salience 1.0 (spec section 6: a ring always passes).

Burst frames keep their own per-frame envelope correlation ids: the frame
router keys pending tasks by them, so a shared id would collide. The push's
correlation id is in each burst frame's "Published frame pointer" log line.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import StreetEventV1, VisionEventBundleItem, VisionEventPayload

from .ring_source import RingPush

DOORBELL_RING_EVENT_TYPE = "doorbell_ring"
DOORBELL_MOTION_EVENT_TYPE = "doorbell_motion"
RING_SALIENCE = 1.0
MOTION_SALIENCE = 0.2
# Ring's own sensor said so: not a model's guess.
RING_CONFIDENCE = 1.0
# Ring motion alerts fire on cars, shadows and trees.
MOTION_CONFIDENCE = 0.5

_EVENT_TYPES = {"ding": DOORBELL_RING_EVENT_TYPE, "motion": DOORBELL_MOTION_EVENT_TYPE}
_NARRATIVES = {
    "ding": "Someone pressed the Ring doorbell button.",
    "motion": "The Ring doorbell reported motion at the front door.",
}


def doorbell_event_type(kind: str) -> str | None:
    return _EVENT_TYPES.get(kind)


def ring_event_id(push: RingPush) -> str:
    """Stable per Ring push: a re-delivered push cannot write a second row
    (vision_events.event_id is the primary key; the sql-writer merges). With
    no Ring id, kind + Ring's created_at second stands in for it."""
    key = push.ring_event_id or f"t{int(push.occurred_at)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"orion:doorbell:{push.kind}:{key}"))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _refs(push: RingPush) -> list[str]:
    refs = [f"correlation:{push.correlation_id}", f"ring_kind:{push.raw_kind}", f"live_view:{push.live_view}"]
    if push.ring_event_id:
        refs.insert(0, f"ring_event:{push.ring_event_id}")
    refs.append(f"occurred_at:{_iso(push.occurred_at)}")
    return refs


def build_vision_event(push: RingPush, *, stream_id: str) -> VisionEventBundleItem | None:
    event_type = doorbell_event_type(push.kind)
    if event_type is None:
        return None
    ring = push.kind == "ding"
    return VisionEventBundleItem(
        event_id=ring_event_id(push),
        event_type=event_type,
        narrative=_NARRATIVES[push.kind],
        entities=[],
        tags=[event_type, "ring_push"],
        confidence=RING_CONFIDENCE if ring else MOTION_CONFIDENCE,
        salience=RING_SALIENCE if ring else MOTION_SALIENCE,
        evidence_refs=_refs(push),
        stream_id=stream_id,
    )


def build_street_event(push: RingPush, *, stream_id: str) -> StreetEventV1 | None:
    if push.kind != "ding":
        return None
    return StreetEventV1(
        event_id=ring_event_id(push),
        kind="doorbell_ring",
        stream_ids=[stream_id],
        occurred_at=datetime.fromtimestamp(push.occurred_at, tz=timezone.utc),
        subject_kind="person",  # a button press needs a hand
        salience=RING_SALIENCE,
        evidence_refs=_refs(push),
        correlation_id=push.correlation_id,
    )


def vision_event_envelope(push: RingPush, item: VisionEventBundleItem, *, service_name: str, service_version: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="vision.event.bundle",
        source=ServiceRef(name=service_name, version=service_version),
        correlation_id=uuid.UUID(push.correlation_id),
        payload=VisionEventPayload(events=[item]).model_dump(mode="json"),
    )


def street_event_envelope(push: RingPush, event: StreetEventV1, *, service_name: str, service_version: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="vision.street.event.v1",
        source=ServiceRef(name=service_name, version=service_version),
        correlation_id=uuid.UUID(push.correlation_id),
        payload=event.model_dump(mode="json"),
    )
