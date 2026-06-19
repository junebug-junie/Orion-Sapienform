"""Tests for hub chat grammar publisher — fail-open semantics."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1


def _make_event(event_kind: str = "trace_started") -> GrammarEventV1:
    return GrammarEventV1(
        event_id="evt-001",
        trace_id="trace-001",
        event_kind=event_kind,  # type: ignore[arg-type]
        emitted_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        provenance=GrammarProvenanceV1(source_service="orion-hub"),
    )


# ---------------------------------------------------------------------------
# 1. Enabled + one event → publish_grammar_event called once
# ---------------------------------------------------------------------------

def test_publish_enabled_calls_bus():
    from scripts.grammar_publish import publish_hub_chat_grammar_trace

    event = _make_event()
    bus = MagicMock()

    with patch("scripts.grammar_publish.publish_grammar_event", new_callable=AsyncMock) as mock_pub:
        asyncio.run(
            publish_hub_chat_grammar_trace(
                bus,
                [event],
                correlation_id="corr-1",
                channel="orion:grammar:event",
                enabled=True,
            )
        )

    mock_pub.assert_awaited_once_with(
        bus,
        event,
        source_name="orion-hub",
        channel="orion:grammar:event",
    )


# ---------------------------------------------------------------------------
# 2. enabled=False → no bus calls
# ---------------------------------------------------------------------------

def test_publish_disabled_is_noop():
    from scripts.grammar_publish import publish_hub_chat_grammar_trace

    event = _make_event()
    bus = MagicMock()

    with patch("scripts.grammar_publish.publish_grammar_event", new_callable=AsyncMock) as mock_pub:
        asyncio.run(
            publish_hub_chat_grammar_trace(
                bus,
                [event],
                correlation_id="corr-2",
                channel="orion:grammar:event",
                enabled=False,
            )
        )

    mock_pub.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. Empty events list → no bus calls
# ---------------------------------------------------------------------------

def test_publish_empty_events_is_noop():
    from scripts.grammar_publish import publish_hub_chat_grammar_trace

    bus = MagicMock()

    with patch("scripts.grammar_publish.publish_grammar_event", new_callable=AsyncMock) as mock_pub:
        asyncio.run(
            publish_hub_chat_grammar_trace(
                bus,
                [],
                correlation_id="corr-3",
                channel="orion:grammar:event",
                enabled=True,
            )
        )

    mock_pub.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Bus raises RuntimeError → exception does not propagate out
# ---------------------------------------------------------------------------

def test_publish_bus_error_does_not_raise():
    from scripts.grammar_publish import publish_hub_chat_grammar_trace

    event = _make_event()
    bus = MagicMock()

    with patch(
        "scripts.grammar_publish.publish_grammar_event",
        new_callable=AsyncMock,
        side_effect=RuntimeError("bus exploded"),
    ):
        # Must not raise
        asyncio.run(
            publish_hub_chat_grammar_trace(
                bus,
                [event],
                correlation_id="corr-4",
                channel="orion:grammar:event",
                enabled=True,
            )
        )


# ---------------------------------------------------------------------------
# 5. First event raises, second event is still attempted
# ---------------------------------------------------------------------------

def test_publish_per_event_error_continues():
    from scripts.grammar_publish import publish_hub_chat_grammar_trace

    event_a = _make_event("trace_started")
    event_b = _make_event("trace_ended")
    bus = MagicMock()

    call_count = 0

    async def _side_effect(bus, event, *, source_name, channel):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first event fails")

    with patch("scripts.grammar_publish.publish_grammar_event", side_effect=_side_effect):
        asyncio.run(
            publish_hub_chat_grammar_trace(
                bus,
                [event_a, event_b],
                correlation_id="corr-5",
                channel="orion:grammar:event",
                enabled=True,
            )
        )

    assert call_count == 2, "Both events must be attempted even if first raises"
