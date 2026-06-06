from datetime import datetime, timezone
import pytest
from orion.schemas.graph_compression import (
    CompressionRegionV1,
    CompressionStalenessMarkV1,
    GraphCompressionRegionMaterializedV1,
)


def test_compression_region_v1_round_trip():
    now = datetime.now(timezone.utc)
    r = CompressionRegionV1(
        region_id="urn:orion:compression:region:abc123",
        scope="episodic",
        kind="community",
        summary="A cluster of memory fragments about workflow design.",
        summary_kind="llm",
        salience=0.72,
        trust_tier="verified",
        exemplar_ids=["http://conjourney.net/chat/turn/1"],
        derived_from=["http://conjourney.net/chat/turn/1"],
        generated_at=now,
        compression_version="1.0.0",
    )
    data = r.model_dump(mode="json")
    restored = CompressionRegionV1.model_validate(data)
    assert restored.region_id == r.region_id
    assert restored.scope == "episodic"
    assert restored.stale is False


def test_compression_region_v1_requires_exemplar_ids():
    with pytest.raises(Exception):
        CompressionRegionV1(
            region_id="urn:orion:compression:region:abc",
            scope="episodic",
            kind="community",
            summary="x",
            summary_kind="structural",
            salience=0.5,
            trust_tier="unverified",
            exemplar_ids=[],  # must be non-empty per spec
            derived_from=["x"],
            generated_at=datetime.now(timezone.utc),
            compression_version="1.0.0",
        )


def test_staleness_mark_v1_round_trip():
    import time
    m = CompressionStalenessMarkV1(
        scope="episodic",
        reason="rdf_enqueue_trigger",
        source_service="orion-rdf-writer",
        ts=time.time(),
    )
    assert m.region_id is None  # scope-wide mark


def test_materialized_v1_round_trip():
    import time
    e = GraphCompressionRegionMaterializedV1(
        region_id="urn:orion:compression:region:abc123",
        scope="episodic",
        kind="community",
        salience=0.72,
        trust_tier="verified",
        summary_kind="llm",
        compression_version="1.0.0",
        ts=time.time(),
    )
    data = e.model_dump(mode="json")
    assert data["region_id"] == e.region_id
