from __future__ import annotations

from typing import Any

from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope


async def publish_with_reconnect(
    bus: OrionBusAsync,
    channel: str,
    msg: BaseEnvelope | dict[str, Any],
    *,
    log_label: str = "bus_publish",
) -> None:
    """Publish once; on transport failure reconnect the command client and retry."""
    try:
        await bus.publish(channel, msg)
    except Exception as exc:
        logger.warning("{} failed channel={} err={}; reconnecting", log_label, channel, exc)
        await bus.reconnect()
        await bus.publish(channel, msg)
