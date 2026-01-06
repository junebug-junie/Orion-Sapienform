from __future__ import annotations

import time
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.pad import PadEventV1, PadLinks


async def metric_reducer(env: BaseEnvelope, *, channel: str) -> PadEventV1:
    payload = env.payload if isinstance(env.payload, dict) else {}
    created_ts = int(env.created_at.timestamp() * 1000) if env.created_at else int(time.time() * 1000)
    metric_name = payload.get("metric") or payload.get("name") or env.kind
    links = PadLinks(correlation_id=env.correlation_id, trace_id=str(env.correlation_id))
    return PadEventV1(
        event_id=str(uuid4()),
        ts_ms=created_ts,
        source_service=env.source.name if env.source else "unknown",
        source_channel=channel,
        subject=str(metric_name),
        type="metric",
        salience=float(payload.get("salience") or 0.2),
        confidence=float(payload.get("confidence") or 0.4),
        novelty=float(payload.get("novelty") or 0.2),
        payload=payload,
        links=links,
    )


async def snapshot_reducer(env: BaseEnvelope, *, channel: str) -> PadEventV1:
    payload = env.payload if isinstance(env.payload, dict) else {}
    created_ts = int(env.created_at.timestamp() * 1000) if env.created_at else int(time.time() * 1000)
    subject = payload.get("source_node") or payload.get("node") or payload.get("subject")
    links = PadLinks(correlation_id=env.correlation_id, trace_id=str(env.correlation_id))
    return PadEventV1(
        event_id=str(uuid4()),
        ts_ms=created_ts,
        source_service=env.source.name if env.source else "unknown",
        source_channel=channel,
        subject=str(subject) if subject else None,
        type="snapshot",
        salience=float(payload.get("salience") or 0.3),
        confidence=float(payload.get("confidence") or 0.5),
        novelty=float(payload.get("novelty") or 0.2),
        payload=payload,
        links=links,
    )
