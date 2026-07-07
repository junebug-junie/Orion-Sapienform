from __future__ import annotations

import logging
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.contracts.memory_cards import MemoryCardV1, MemoryPriority

logger = logging.getLogger(__name__)

MEMORY_CARD_ACTIVE_CHANNEL = "orion:memory:cards:active"
MEMORY_CARD_ACTIVE_KIND = "memory.card.active.v1"

CRYSTALLIZER_CARD_PRIORITIES: frozenset[MemoryPriority] = frozenset({"high_recall", "always_inject"})


def card_qualifies_for_crystallizer(card: MemoryCardV1) -> bool:
    return card.status == "active" and card.priority in CRYSTALLIZER_CARD_PRIORITIES


def _source(service_name: str, *, version: str = "0.1.0", node: str = "hub") -> ServiceRef:
    return ServiceRef(name=service_name, version=version, node=node)


async def emit_memory_card_active_for_crystallizer(
    bus: Optional[OrionBusAsync],
    card: MemoryCardV1,
    *,
    service_name: str = "orion-hub",
    service_version: str = "0.1.0",
    node_name: str = "hub",
) -> bool:
    """Notify crystallizer when an active card has high-salience priority."""
    if not card_qualifies_for_crystallizer(card):
        return False
    if bus is None or not getattr(bus, "enabled", True):
        logger.debug(
            "memory_card_active_emit_skipped bus_unavailable card_id=%s",
            card.card_id,
        )
        return False

    env = BaseEnvelope(
        kind=MEMORY_CARD_ACTIVE_KIND,
        source=_source(service_name, version=service_version, node=node_name),
        payload=card.model_dump(mode="json"),
    )
    try:
        await bus.publish(MEMORY_CARD_ACTIVE_CHANNEL, env)
        logger.info(
            "memory_card_active_emitted channel=%s card_id=%s priority=%s",
            MEMORY_CARD_ACTIVE_CHANNEL,
            card.card_id,
            card.priority,
        )
        return True
    except Exception as exc:
        logger.warning(
            "memory_card_active_emit_failed card_id=%s error=%s",
            card.card_id,
            exc,
        )
        return False
