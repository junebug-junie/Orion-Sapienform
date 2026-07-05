from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orion.harness.grammar_publish import publish_harness_step_grammar
from orion.schemas.thought import GrammarReceiptV1


@pytest.mark.asyncio
async def test_harness_grammar_publish_per_step() -> None:
    bus = AsyncMock()
    receipts: list[GrammarReceiptV1] = []

    async def _capture(event, **kwargs):
        receipts.append(
            GrammarReceiptV1(
                step_index=kwargs["step_index"],
                tool_name=kwargs.get("tool_name"),
                summary=kwargs["summary"],
                grammar_event_id=event.event_id,
            )
        )

    await publish_harness_step_grammar(
        bus,
        correlation_id="corr-1",
        channel="orion:grammar:event",
        step_index=0,
        tool_name="Read",
        summary="read file",
        publish_fn=_capture,
    )
    assert len(receipts) == 1
    assert receipts[0].tool_name == "Read"
    assert receipts[0].grammar_event_id
