from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from orion.grammar.publish import publish_grammar_event
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("orion.cortex.exec.grammar_publish")


async def publish_cortex_exec_grammar_trace(
    bus: Any,
    events: list[GrammarEventV1],
    *,
    correlation_id: str,
    channel: str,
    source_name: str = "orion-cortex-exec",
    enabled: bool = True,
) -> None:
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
                "cortex_exec_grammar_publish_failed corr=%s event_kind=%s",
                correlation_id,
                event.event_kind,
                exc_info=True,
            )


async def flush_cortex_exec_grammar(
    bus: Any,
    collector: Any | None,
    *,
    correlation_id: str,
    channel: str,
    source_name: str,
    enabled: bool,
) -> None:
    from .grammar_emit import build_cortex_exec_grammar_events

    if collector is None or not enabled:
        return
    events = build_cortex_exec_grammar_events(collector)
    await publish_cortex_exec_grammar_trace(
        bus,
        events,
        correlation_id=correlation_id,
        channel=channel,
        source_name=source_name,
        enabled=enabled,
    )
