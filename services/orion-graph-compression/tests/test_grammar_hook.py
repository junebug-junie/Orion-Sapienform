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

    mock_bus.publish.assert_called_once()
    call_args = mock_bus.publish.call_args
    channel = call_args[0][0]
    assert channel == "orion:substrate:mutation:pressure"
    envelope = call_args[0][1]
    pressure = envelope.payload
    assert pressure.get("source_service") == "orion-graph-compression"
    assert "contradiction" in str(pressure.get("pressure_category", ""))


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
        kind="community",  # not contradiction
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
    mock_bus.publish.assert_not_called()
