import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone


def _make_contradiction_region():
    from orion.schemas.graph_compression import CompressionRegionV1
    return CompressionRegionV1(
        region_id="urn:orion:compression:region:contradiction1",
        scope="substrate",
        kind="contradiction",
        summary="Conflicting beliefs about X and Y.",
        summary_kind="llm",
        salience=0.85,
        trust_tier="unverified",
        exemplar_ids=["http://conjourney.net/substrate/node/1"],
        derived_from=["http://conjourney.net/substrate/node/1"],
        generated_at=datetime.now(timezone.utc),
        compression_version="1.0.0",
    )


def test_contradiction_region_emits_mutation_pressure():
    """Writing a contradiction region must publish MutationPressureEvidenceV1 on the pressure channel."""
    from app.writer import CompressionWriter

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    w = CompressionWriter(
        update_url="http://fuseki/update",
        user="admin",
        password="orion",
        timeout_sec=5.0,
        bus=mock_bus,
        service_name="orion-graph-compression",
        service_version="0.1.0",
        channel_events="orion:graph:compression:events",
        channel_pressure="orion:substrate:mutation:pressure",
    )

    region = _make_contradiction_region()

    async def run():
        await w._emit_grammar_hook(region)

    asyncio.run(run())

    mock_bus.publish.assert_called()
    # Find the call to the pressure channel
    pressure_call = None
    for call in mock_bus.publish.call_args_list:
        if call[0][0] == "orion:substrate:mutation:pressure":
            pressure_call = call
            break
    assert pressure_call is not None, "No publish call to pressure channel"
    channel = pressure_call[0][0]
    assert channel == "orion:substrate:mutation:pressure"
    envelope = pressure_call[0][1]
    pressure = envelope.payload
    assert pressure.get("source_service") == "orion-graph-compression"
    assert pressure.get("pressure_category") == "unsupported_memory_claim"
    assert "contradiction" in str(pressure.get("metadata", {}).get("compression_kind", ""))


def test_non_contradiction_region_does_not_emit_pressure():
    from app.writer import CompressionWriter

    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock()

    w = CompressionWriter(
        update_url="http://fuseki/update",
        user="admin",
        password="orion",
        timeout_sec=5.0,
        bus=mock_bus,
        service_name="orion-graph-compression",
        service_version="0.1.0",
        channel_events="orion:graph:compression:events",
        channel_pressure="orion:substrate:mutation:pressure",
    )

    from orion.schemas.graph_compression import CompressionRegionV1
    region = CompressionRegionV1(
        region_id="urn:orion:compression:region:community1",
        scope="episodic",
        kind="community",
        summary="Normal community.",
        summary_kind="structural",
        salience=0.5,
        trust_tier="unverified",
        exemplar_ids=["http://example.org/1"],
        derived_from=["http://example.org/1"],
        generated_at=datetime.now(timezone.utc),
        compression_version="1.0.0",
    )

    async def run():
        await w._emit_grammar_hook(region)

    asyncio.run(run())
    # Only the materialization event should be published, not the pressure channel
    for call in mock_bus.publish.call_args_list:
        assert call[0][0] != "orion:substrate:mutation:pressure", "Should not emit pressure for non-contradiction"
