"""Cortex chat turn -> attention schema surface (app/attention_schema_publish.py).

Fail-open contract: unbound bus is a warning and False, a raising bus is a
warning and False, and a bound bus gets exactly one AttentionSchemaV1
envelope on orion:attention:schema with process="cortex_turn".
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app import attention_schema_publish as asp
from orion.schemas.attention_frame import AttentionFrameV1, CuriosityCandidateActionV1, OpenLoopV1
from orion.schemas.attention_schema import ATTENTION_SCHEMA_CHANNEL, ATTENTION_SCHEMA_KIND


def _frame() -> AttentionFrameV1:
    return AttentionFrameV1(
        generated_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
        turn_id="turn-1",
        correlation_id="corr-1",
        open_loops=[OpenLoopV1(id="loop-1", description="Zephyr Bridge")],
        selected_action=CuriosityCandidateActionV1(action_type="watch", open_loop_id="loop-1", score=0.62, rationale="keep an eye on it"),
    )


@pytest.fixture(autouse=True)
def _reset():
    asp.reset_attention_schema_bus_for_tests()
    yield
    asp.reset_attention_schema_bus_for_tests()


@pytest.mark.asyncio
async def test_unbound_bus_is_a_noop_that_reports_false():
    assert await asp.publish_attention_schema(_frame()) is False


@pytest.mark.asyncio
async def test_bound_bus_gets_one_cortex_turn_row():
    bus = AsyncMock()
    asp.bind_attention_schema_bus(bus)
    assert await asp.publish_attention_schema(_frame()) is True
    bus.publish.assert_awaited_once()
    channel, envelope = bus.publish.await_args.args
    assert channel == ATTENTION_SCHEMA_CHANNEL
    assert envelope.kind == ATTENTION_SCHEMA_KIND
    assert envelope.payload["process"] == "cortex_turn"
    assert envelope.payload["entry_id"] == "cortex-turn-1"
    assert envelope.payload["attention_reason"] == "selected:watch"
    assert envelope.payload["reason_narrative"] == "keep an eye on it"
    assert envelope.payload["attended_label"] == "Zephyr Bridge"


@pytest.mark.asyncio
async def test_a_raising_bus_never_propagates():
    bus = AsyncMock()
    bus.publish.side_effect = RuntimeError("bus down")
    asp.bind_attention_schema_bus(bus)
    assert await asp.publish_attention_schema(_frame()) is False
