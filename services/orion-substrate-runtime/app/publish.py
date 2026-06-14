from __future__ import annotations

from typing import Any

from orion.grammar.publish import publish_grammar_event
from orion.schemas.grammar import GrammarEventV1


async def publish_accepted_events(
    bus: Any,
    events: list[GrammarEventV1],
    *,
    channel: str,
) -> None:
    for event in events:
        await publish_grammar_event(
            bus,
            event,
            source_name="orion-substrate-runtime",
            channel=channel,
        )
