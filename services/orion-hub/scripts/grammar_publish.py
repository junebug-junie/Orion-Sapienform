from __future__ import annotations

import logging
from typing import Any

from orion.grammar.publish import publish_grammar_event
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("orion-hub.grammar_publish")


async def publish_hub_chat_grammar_trace(
    bus: Any,
    events: list[GrammarEventV1],
    *,
    correlation_id: str,
    channel: str,
    enabled: bool = True,
) -> None:
    """Fail-open. Chat must work whether grammar publishing is on or off."""
    if not enabled or not events:
        return
    for event in events:
        try:
            await publish_grammar_event(
                bus,
                event,
                source_name="orion-hub",
                channel=channel,
            )
        except Exception:
            logger.warning(
                "hub_chat_grammar_publish_failed corr=%s event_kind=%s",
                correlation_id,
                event.event_kind,
                exc_info=True,
            )
