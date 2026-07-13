"""Bus-publish wrapper for orch route-arbitration grammar events.

Mirrors services/orion-hub/scripts/grammar_publish.py
(publish_hub_chat_grammar_trace) and
services/orion-cortex-exec/app/grammar_publish.py
(publish_cortex_exec_grammar_trace). Fail-open: a publish failure here must
never raise into the caller or affect the chat response.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from orion.grammar.publish import publish_grammar_event
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("orion.cortex.orch.grammar_publish")


async def publish_orch_route_grammar_trace(
    bus: Any,
    events: list[GrammarEventV1],
    *,
    correlation_id: str,
    channel: str,
    source_name: str = "orion-cortex-orch",
    enabled: bool = True,
) -> None:
    """Fire-and-forget shadow publish. No-op unless enabled and non-empty."""
    if not enabled or not events:
        return
    try:
        corr_uuid = UUID(str(correlation_id))
    except (ValueError, TypeError):
        corr_uuid = None
    for event in events:
        try:
            await publish_grammar_event(
                bus,
                event,
                source_name=source_name,
                correlation_id=corr_uuid,
                channel=channel,
            )
        except Exception:
            logger.warning(
                "orch_route_grammar_publish_failed corr=%s event_kind=%s",
                correlation_id,
                event.event_kind,
                exc_info=True,
            )
