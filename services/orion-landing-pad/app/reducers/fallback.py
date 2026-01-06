from __future__ import annotations

import time
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.pad import PadEventV1, PadLinks


async def fallback_reducer(env: BaseEnvelope, *, channel: str) -> PadEventV1:
    created_ts = int(env.created_at.timestamp() * 1000) if env.created_at else int(time.time() * 1000)
    payload = env.payload if isinstance(env.payload, dict) else {"raw": env.payload}
    payload = {"raw": payload, "kind": env.kind}
    links = PadLinks(correlation_id=env.correlation_id, trace_id=str(env.correlation_id))
    return PadEventV1(
        event_id=str(uuid4()),
        ts_ms=created_ts,
        source_service=env.source.name if env.source else "unknown",
        source_channel=channel,
        subject=payload.get("subject"),
        type="unknown",
        salience=0.1,
        confidence=0.25,
        novelty=0.25,
        payload=payload,
        links=links,
    )
