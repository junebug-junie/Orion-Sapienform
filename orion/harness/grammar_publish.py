from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.harness_finalize import GrammarReceiptV1

PublishFn = Callable[..., Awaitable[None]]

logger = logging.getLogger("orion.harness.grammar_publish")


def _build_harness_step_event(
    *,
    correlation_id: str,
    step_index: int,
    tool_name: str | None,
    summary: str,
) -> GrammarEventV1:
    event_id = str(uuid.uuid4())
    trace_id = correlation_id
    clipped_summary = summary[:500]
    tool_label = tool_name or "none"
    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:harness_step:{step_index}",
        trace_id=trace_id,
        atom_type="reasoning_step",
        semantic_role="harness_fcc_step",
        layer="harness",
        dimensions=["harness", "fcc", "step"],
        summary=f"Harness step {step_index}: tool={tool_label}, {clipped_summary}",
        source_event_id=event_id,
    )
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=trace_id,
        correlation_id=correlation_id,
        emitted_at=datetime.now(timezone.utc),
        provenance=GrammarProvenanceV1(
            source_service="orion-harness-governor",
            source_component="grammar_publish",
        ),
        atom=atom,
    )


async def publish_harness_step_grammar(
    bus: Any,
    *,
    correlation_id: str,
    channel: str,
    step_index: int,
    tool_name: str | None,
    summary: str,
    publish_fn: PublishFn | None = None,
) -> GrammarReceiptV1:
    event = _build_harness_step_event(
        correlation_id=correlation_id,
        step_index=step_index,
        tool_name=tool_name,
        summary=summary,
    )
    if publish_fn is not None:
        await publish_fn(event, step_index=step_index, tool_name=tool_name, summary=summary)
    else:
        from orion.grammar.publish import publish_grammar_event

        await publish_grammar_event(
            bus, event, source_name="orion-harness-governor", channel=channel
        )
    logger.info(
        "harness_grammar_step_published corr=%s channel=%s step=%s tool=%s event_id=%s",
        correlation_id,
        channel,
        step_index,
        tool_name or "none",
        event.event_id,
    )
    return GrammarReceiptV1(
        step_index=step_index,
        tool_name=tool_name,
        summary=summary,
        grammar_event_id=event.event_id,
    )
