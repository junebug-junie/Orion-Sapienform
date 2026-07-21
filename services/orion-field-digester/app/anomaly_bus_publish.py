from __future__ import annotations

import logging

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.field_channel_anomaly_score import FieldChannelAnomalyScoreV1

logger = logging.getLogger("orion.field.digester.anomaly_bus_publish")

FIELD_CHANNEL_ANOMALY_SCORE_KIND = "field_channel_anomaly.score.v1"


async def publish_anomaly_score(
    *,
    bus_url: str,
    bus_enabled: bool,
    channel: str,
    score: FieldChannelAnomalyScoreV1,
    service_name: str,
    service_version: str,
    node_name: str,
) -> None:
    """Fresh OrionBusAsync per publish -- this fires at most once per
    FIELD_CHANNEL_ANOMALY_CHECK_INTERVAL_SEC (default 60s), not a hot path,
    same one-shot connect/publish/close pattern as
    orion/substrate/causal_geometry_bus_publish.py. Never raises: called
    from _anomaly_loop(), which already wraps each tick in a broad
    try/except, but this mirrors that module's "every failure degrades,
    never propagates" posture explicitly rather than relying solely on the
    caller."""
    if not bus_enabled:
        return
    bus = OrionBusAsync(url=bus_url, enabled=True)
    try:
        await bus.connect()
        envelope = BaseEnvelope(
            kind=FIELD_CHANNEL_ANOMALY_SCORE_KIND,
            source=ServiceRef(name=service_name, version=service_version, node=node_name),
            correlation_id=score.correlation_id,
            payload=score.model_dump(mode="json"),
        )
        await bus.publish(channel, envelope)
    except Exception:
        logger.warning("field_channel_anomaly_score_publish_failed", exc_info=True)
    finally:
        try:
            await bus.close()
        except Exception:
            pass
