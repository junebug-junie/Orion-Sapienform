"""Publish grammar events to the Orion bus (optional path for emitters)."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.grammar import GrammarEventV1

GRAMMAR_EVENT_CHANNEL = "orion:grammar:event"


async def publish_grammar_event(
    bus: Any,
    event: GrammarEventV1,
    *,
    source_name: str = "orion-grammar",
    correlation_id: UUID | None = None,
    channel: str | None = None,
) -> None:
    """Publish a ``grammar.event.v1`` envelope on the grammar event channel."""
    corr = correlation_id
    if corr is None and event.correlation_id:
        try:
            corr = UUID(str(event.correlation_id))
        except (ValueError, TypeError):
            corr = uuid4()
    if corr is None:
        corr = uuid4()

    envelope = BaseEnvelope(
        kind="grammar.event.v1",
        source=ServiceRef(name=source_name),
        correlation_id=corr,
        payload=event.model_dump(mode="json"),
    )
    await bus.publish(channel or GRAMMAR_EVENT_CHANNEL, envelope)
