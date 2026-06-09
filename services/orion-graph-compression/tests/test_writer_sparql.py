import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone


def _make_region(kind="community"):
    from orion.schemas.graph_compression import CompressionRegionV1
    return CompressionRegionV1(
        region_id="urn:orion:compression:region:abc123",
        scope="episodic",
        kind=kind,
        summary="Test summary.",
        summary_kind="structural",
        salience=0.7,
        trust_tier="unverified",
        exemplar_ids=["http://conjourney.net/chat/turn/1"],
        derived_from=["http://conjourney.net/chat/turn/1"],
        generated_at=datetime.now(timezone.utc),
        compression_version="1.0.0",
    )


def test_writer_builds_sparql_update_targeting_compressions_graph():
    from app.writer import CompressionWriter
    w = CompressionWriter(
        update_url="http://fuseki/update",
        user="admin",
        password="orion",
        timeout_sec=5.0,
        bus=None,
        service_name="orion-graph-compression",
        service_version="0.1.0",
        channel_events="orion:graph:compression:events",
        channel_pressure="orion:substrate:mutation:pressure",
    )
    region = _make_region()
    sparql = w._build_sparql_update(region)
    assert "orion/compressions" in sparql
    assert region.region_id in sparql
    assert "INSERT DATA" in sparql


def test_writer_includes_summary_literal():
    from app.writer import CompressionWriter
    w = CompressionWriter(
        update_url="http://fuseki/update",
        user="admin",
        password="orion",
        timeout_sec=5.0,
        bus=None,
        service_name="orion-graph-compression",
        service_version="0.1.0",
        channel_events="orion:graph:compression:events",
        channel_pressure="orion:substrate:mutation:pressure",
    )
    region = _make_region()
    sparql = w._build_sparql_update(region)
    assert "Test summary." in sparql
