from __future__ import annotations

import logging
from typing import Any, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.memory.crystallization.schemas import ActiveMemoryPacketV1, MemoryCrystallizationV1

logger = logging.getLogger(__name__)

LIFECYCLE_KINDS = {
    "proposed": "memory.crystallization.proposed.v1",
    "validated": "memory.crystallization.validated.v1",
    "approved": "memory.crystallization.approved.v1",
    "rejected": "memory.crystallization.rejected.v1",
    "quarantined": "memory.crystallization.quarantined.v1",
    "project": "memory.crystallization.project.v1",
    "retrieved": "memory.crystallization.retrieved.v1",
}

CHANNEL_DEFAULTS = {
    "proposed": "orion:memory:crystallization:proposed",
    "validated": "orion:memory:crystallization:validated",
    "approved": "orion:memory:crystallization:approved",
    "rejected": "orion:memory:crystallization:rejected",
    "quarantined": "orion:memory:crystallization:quarantined",
    "project": "orion:memory:crystallization:project",
    "retrieved": "orion:memory:crystallization:retrieved",
}


def _source(service_name: str, *, version: str = "0.1.0", node: str = "hub") -> ServiceRef:
    return ServiceRef(name=service_name, version=version, node=node)


async def emit_crystallization_lifecycle(
    bus: Optional[OrionBusAsync],
    *,
    lifecycle: str,
    crystallization: MemoryCrystallizationV1,
    service_name: str = "orion-hub",
    service_version: str = "0.1.0",
    node_name: str = "hub",
    channel: str | None = None,
) -> bool:
    if bus is None or not getattr(bus, "enabled", True):
        logger.debug("bus_unavailable lifecycle=%s crystallization_id=%s", lifecycle, crystallization.crystallization_id)
        return False

    kind = LIFECYCLE_KINDS.get(lifecycle)
    if not kind:
        logger.warning("unknown_crystallization_lifecycle lifecycle=%s", lifecycle)
        return False

    target_channel = channel or CHANNEL_DEFAULTS.get(lifecycle, "")
    if not target_channel:
        return False

    env = BaseEnvelope(
        kind=kind,
        source=_source(service_name, version=service_version, node=node_name),
        payload=crystallization.model_dump(mode="json"),
    )
    try:
        await bus.publish(target_channel, env)
        logger.info(
            "crystallization_lifecycle_emitted lifecycle=%s channel=%s id=%s",
            lifecycle,
            target_channel,
            crystallization.crystallization_id,
        )
        return True
    except Exception as exc:
        logger.warning(
            "crystallization_lifecycle_emit_failed lifecycle=%s id=%s error=%s",
            lifecycle,
            crystallization.crystallization_id,
            exc,
        )
        return False


async def emit_active_packet_retrieved(
    bus: Optional[OrionBusAsync],
    packet: ActiveMemoryPacketV1,
    *,
    service_name: str = "orion-hub",
    service_version: str = "0.1.0",
    node_name: str = "hub",
    channel: str = "orion:memory:crystallization:retrieved",
) -> bool:
    if bus is None:
        return False
    env = BaseEnvelope(
        kind=LIFECYCLE_KINDS["retrieved"],
        source=_source(service_name, version=service_version, node=node_name),
        payload=packet.model_dump(mode="json"),
    )
    try:
        await bus.publish(channel, env)
        return True
    except Exception as exc:
        logger.warning("active_packet_emit_failed error=%s", exc)
        return False


async def emit_vector_upsert(
    bus: Optional[OrionBusAsync],
    *,
    payload: dict[str, Any],
    channel: str = "orion:memory:vector:upsert",
    service_name: str = "orion-hub",
    service_version: str = "0.1.0",
    node_name: str = "hub",
) -> bool:
    if bus is None:
        return False
    env = BaseEnvelope(
        kind="memory.vector.upsert.v1",
        source=_source(service_name, version=service_version, node=node_name),
        payload=payload,
    )
    try:
        await bus.publish(channel, env)
        return True
    except Exception as exc:
        logger.warning("vector_upsert_emit_failed error=%s", exc)
        return False
