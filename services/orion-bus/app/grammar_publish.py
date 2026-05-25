from __future__ import annotations

import logging
from typing import Any

from orion.grammar.publish import publish_grammar_event
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("orion.bus.grammar_publish")


async def publish_bus_transport_grammar_trace(
    bus: Any,
    events: list[GrammarEventV1],
    *,
    channel: str,
    source_name: str = "orion-bus",
    enabled: bool = True,
) -> None:
    if not enabled or not events:
        return
    for event in events:
        try:
            await publish_grammar_event(
                bus,
                event,
                source_name=source_name,
                correlation_id=None,
                channel=channel,
            )
        except Exception:
            logger.warning(
                "bus_grammar_publish_failed trace_id=%s event_kind=%s",
                event.trace_id,
                event.event_kind,
                exc_info=True,
            )
