"""Unit tests for OrionBusAsync._emit_rpc_timeout_grammar (Option C of
docs/superpowers/specs/2026-07-24-transport-metacog-trigger-design.md).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orion.core.bus.async_service import OrionBusAsync


def _bus() -> OrionBusAsync:
    bus = OrionBusAsync("redis://localhost:6379/0", enabled=False)
    bus.publish = AsyncMock()
    return bus


@pytest.mark.asyncio
async def test_emits_real_grammar_atom_with_correct_semantic_role() -> None:
    bus = _bus()
    await bus._emit_rpc_timeout_grammar(
        request_channel="orion:cortex:exec:request:background",
        reply_channel="orion:exec:result:xyz",
        corr="c0ffee00-0000-4000-8000-000000000001",
        timeout_sec=60.0,
        timeout_elapsed_ms=60123.4,
    )
    assert bus.publish.await_count == 1
    channel, envelope = bus.publish.await_args.args
    assert channel == "orion:grammar:event"
    payload = envelope.payload
    assert payload["event_kind"] == "atom_emitted"
    assert payload["atom"]["semantic_role"] == "rpc_transport_timeout"
    assert payload["atom"]["text_value"] == "orion:cortex:exec:request:background"
    assert payload["correlation_id"] == "c0ffee00-0000-4000-8000-000000000001"
    assert payload["provenance"]["source_service"] == "orion-bus"
    assert payload["trace_id"].startswith("bus.transport:rpc_timeout:")


@pytest.mark.asyncio
async def test_never_raises_when_publish_fails() -> None:
    """Fire-and-forget guarantee -- a grammar-publish failure must never surface
    as an exception from the timeout branch that's already handling a real
    RPC failure."""
    bus = _bus()
    bus.publish = AsyncMock(side_effect=RuntimeError("redis down"))
    await bus._emit_rpc_timeout_grammar(
        request_channel="orion:cortex:exec:request",
        reply_channel="orion:exec:result:abc",
        corr="c0ffee00-0000-4000-8000-000000000002",
        timeout_sec=30.0,
        timeout_elapsed_ms=30001.0,
    )
    # No exception raised -- reaching this line is the assertion.


@pytest.mark.asyncio
async def test_two_calls_produce_distinct_event_ids() -> None:
    bus = _bus()
    await bus._emit_rpc_timeout_grammar(
        request_channel="orion:a", reply_channel="orion:b",
        corr="c0ffee00-0000-4000-8000-000000000003",
        timeout_sec=10.0, timeout_elapsed_ms=10000.0,
    )
    await bus._emit_rpc_timeout_grammar(
        request_channel="orion:a", reply_channel="orion:b",
        corr="c0ffee00-0000-4000-8000-000000000003",
        timeout_sec=10.0, timeout_elapsed_ms=10000.0,
    )
    assert bus.publish.await_count == 2
    e1 = bus.publish.await_args_list[0].args[1].payload["event_id"]
    e2 = bus.publish.await_args_list[1].args[1].payload["event_id"]
    assert e1 != e2
