from __future__ import annotations

import logging
import time
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.pad import PadEventV1, PadLinks
from orion.schemas.telemetry.spark import SparkStateSnapshotV1

logger = logging.getLogger("orion.landing_pad.reducers")


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
    snap = None
    if isinstance(payload, dict):
        allowed = set(SparkStateSnapshotV1.model_fields.keys())
        filtered = {k: payload[k] for k in payload.keys() if k in allowed}
        try:
            snap = SparkStateSnapshotV1.model_validate(filtered)
        except Exception:
            logger.warning(
                "LANDING_PAD_SPARK_SNAPSHOT_PARSE_FAILED keys=%s",
                list(payload.keys())[:25],
            )
            snap = None
    created_dt = snap.snapshot_ts if snap else env.created_at
    created_ts = int(created_dt.timestamp() * 1000) if created_dt else int(time.time() * 1000)
    subject = (
        snap.source_node
        if snap and snap.source_node
        else payload.get("source_node") or payload.get("node") or payload.get("subject")
    )
    payload = dict(payload)
    if snap:
        payload.setdefault("spark_seq", snap.seq)
        payload.setdefault("spark_snapshot_ts", snap.snapshot_ts.isoformat())
    links = PadLinks(correlation_id=env.correlation_id, trace_id=str(env.correlation_id))
    novelty = payload.get("novelty")
    if novelty is None and snap:
        novelty = (snap.phi or {}).get("novelty")
    return PadEventV1(
        event_id=str(uuid4()),
        ts_ms=created_ts,
        source_service=env.source.name if env.source else "unknown",
        source_channel=channel,
        subject=str(subject) if subject else None,
        type="snapshot",
        salience=float(payload.get("salience") or 0.3),
        confidence=float(payload.get("confidence") or 0.5),
        novelty=float(novelty or 0.2),
        payload=payload,
        links=links,
    )


async def biometrics_reducer(env: BaseEnvelope, *, channel: str) -> PadEventV1:
    payload = env.payload if isinstance(env.payload, dict) else {}
    created_ts = int(env.created_at.timestamp() * 1000) if env.created_at else int(time.time() * 1000)
    composites = payload.get("composites") or {}
    strain = float(composites.get("strain") or 0.0)
    links = PadLinks(correlation_id=env.correlation_id, trace_id=str(env.correlation_id))
    subject = payload.get("node") or payload.get("subject")
    return PadEventV1(
        event_id=str(uuid4()),
        ts_ms=created_ts,
        source_service=env.source.name if env.source else "unknown",
        source_channel=channel,
        subject=str(subject) if subject else None,
        type="biometrics",
        salience=max(0.1, min(1.0, strain)),
        confidence=float(payload.get("confidence") or 0.6),
        novelty=float(payload.get("novelty") or 0.2),
        payload=payload,
        links=links,
    )
