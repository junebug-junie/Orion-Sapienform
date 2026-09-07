from __future__ import annotations

import uuid
from typing import Any

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cockpit_sighting import COCKPIT_HOP_CHANNEL, CockpitHopV1


def _envelope_correlation_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        return uuid.uuid4()


async def publish_cockpit_hop(
    bus: Any,
    hop: CockpitHopV1,
    *,
    channel: str = COCKPIT_HOP_CHANNEL,
) -> None:
    envelope = BaseEnvelope(
        kind="cockpit.hop.v1",
        source=ServiceRef(name="orion-hub"),
        correlation_id=_envelope_correlation_id(hop.correlation_id),
        payload=hop.model_dump(mode="json"),
    )
    await bus.publish(channel, envelope)
