from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.grammar_emit import BusTransportGrammarCollector, build_bus_transport_grammar_events
from app.grammar_publish import publish_bus_transport_grammar_trace


@pytest.mark.asyncio
async def test_publish_failure_is_non_fatal() -> None:
    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    collector = BusTransportGrammarCollector(
        node_id="athena",
        sample_window_id="20260525T170000Z",
        observed_at=datetime.now(timezone.utc),
        code_version="0.1.0",
    )
    collector.record_tick_started()
    collector.record_health_observed(redis_ping_ok=True)
    collector.record_tick_completed(streams_observed=0)
    events = build_bus_transport_grammar_events(collector)
    await publish_bus_transport_grammar_trace(
        bus,
        events,
        channel="orion:grammar:event",
        source_name="orion-bus",
        enabled=True,
    )
