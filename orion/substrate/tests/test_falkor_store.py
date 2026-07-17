from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.falkor_store import (
    FalkorSubstrateStore,
    FalkorSubstrateStoreConfig,
    RecordingFalkorClient,
    RedisGraphQueryClient,
    _normalize_rows,
)
from orion.substrate.graphdb_store import build_substrate_store_from_env
from orion.substrate.store import InMemorySubstrateGraphStore


def _concept(*, node_id: str = "sub-node-a", metadata: dict | None = None) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        label="alpha",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_falkor_store",
            evidence_refs=[],
        ),
        metadata=metadata or {},
    )


def test_falkor_upsert_round_trip_via_cache():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )
    node = _concept()
    store.upsert_node(identity_key="id-a", node=node)
    assert store.get_node_by_id(node.node_id) is not None
    assert store.get_node_id_by_identity("id-a") == node.node_id
    snap = store.snapshot()
    assert node.node_id in snap.nodes
    assert any("MERGE (n:SubstrateNode" in cypher for cypher, _ in client.calls)


def test_falkor_upsert_concept_uses_native_cypher_properties():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    node = _concept(node_id="concept-native-alpha")

    store.upsert_node(identity_key="concept:alpha", node=node)

    cypher, params = client.calls[-1]
    assert "payload_json" not in cypher
    assert params is not None
    assert "payload_json" not in params
    assert "MERGE (n:SubstrateNode:Concept {node_id: $node_id})" in cypher
    assert "n.label = $label" in cypher
    assert "n.promotion_state = $promotion_state" in cypher
    assert "n.salience = $salience" in cypher
    assert params["node_id"] == "concept-native-alpha"
    assert params["node_kind"] == "concept"
    assert params["identity_key"] == "concept:alpha"
    assert params["label"] == "alpha"
    assert params["anchor_scope"] == "orion"
    assert params["promotion_state"] == "proposed"
    assert params["risk_tier"] == "low"
    assert params["salience"] == 0.0
    assert params["activation"] == 0.0
    assert params["recency_score"] == 0.0
    assert params["confidence"] == 0.5


def test_falkor_hydrates_concept_from_native_properties():
    client = RecordingFalkorClient(
        hydrate_rows=[
            {
                "node_id": "concept-hydrated",
                "node_kind": "concept",
                "identity_key": "concept:hydrated",
                "label": "Hydrated concept",
                "definition": "Loaded from Falkor native properties",
                "anchor_scope": "orion",
                "subject_ref": None,
                "promotion_state": "canonical",
                "risk_tier": "low",
                "confidence": 0.75,
                "salience": 0.66,
                "activation": 0.44,
                "recency_score": 0.33,
                "decay_floor": 0.1,
                "decay_half_life_seconds": None,
                "observed_at": "2026-07-16T00:00:00+00:00",
                "valid_from": None,
                "valid_to": None,
                "provenance_authority": "local_inferred",
                "provenance_source_kind": "test",
                "provenance_source_channel": "test:falkor",
                "provenance_producer": "test_falkor_store",
                "provenance_model_name": None,
                "provenance_correlation_id": None,
                "provenance_trace_id": None,
                "provenance_tier_rank": None,
            }
        ]
    )

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    node = store.get_node_by_id("concept-hydrated")
    assert node is not None
    assert node.node_kind == "concept"
    assert node.label == "Hydrated concept"
    assert node.definition == "Loaded from Falkor native properties"
    assert node.promotion_state == "canonical"
    assert node.signals.confidence == 0.75
    assert node.signals.salience == 0.66
    assert node.signals.activation.activation == 0.44
    assert node.signals.activation.recency_score == 0.33
    assert store.get_node_id_by_identity("concept:hydrated") == "concept-hydrated"


def test_falkor_sanitizes_metadata_cathedral():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379"),
        client=client,
        hydrate=False,
    )
    meta = {f"k{i}": i for i in range(20)}
    node = _concept(metadata=meta)
    store.upsert_node(identity_key="id-b", node=node)
    stored = store.get_node_by_id(node.node_id)
    assert stored is not None
    assert len(stored.metadata) <= 16


def test_falkor_edge_is_persisted_as_typed_relationship():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379"),
        client=client,
        hydrate=False,
    )
    edge = SubstrateEdgeV1(
        edge_id="sub-edge-a",
        source=NodeRefV1(node_id="sub-node-a", node_kind="concept"),
        target=NodeRefV1(node_id="sub-node-b", node_kind="concept"),
        predicate="supports",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_falkor_store",
        ),
    )

    store.upsert_edge(identity_key="edge-a", edge=edge)

    cypher, params = client.calls[-1]
    assert "MERGE (source)-[e:`supports`" in cypher
    assert "e.substrate_edge = true" in cypher
    assert params is not None
    assert params["source_id"] == "sub-node-a"
    assert params["target_id"] == "sub-node-b"


def test_build_substrate_store_falkor(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")
    monkeypatch.setenv("FALKORDB_URI", "redis://orion-athena-falkordb:6379")
    monkeypatch.setenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")

    # Avoid opening a real redis connection during construction hydrate.
    from orion.substrate import falkor_store as fs

    fake = RecordingFalkorClient()

    class _NoRedis(FalkorSubstrateStore):
        def __init__(self, cfg, *, client=None, hydrate=True):
            super().__init__(cfg, client=client or fake, hydrate=False)

    monkeypatch.setattr(fs, "FalkorSubstrateStore", _NoRedis)
    store = build_substrate_store_from_env()
    assert isinstance(store, FalkorSubstrateStore)


def test_build_substrate_store_falkor_missing_uri(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    store = build_substrate_store_from_env()
    assert isinstance(store, InMemorySubstrateGraphStore)


def test_redis_graph_client_uses_redis_py_parameter_header():
    class _Result:
        result_set = [["payload", "identity"]]

    class _Graph:
        def __init__(self):
            self.calls = []

        def query(self, cypher, params=None):
            self.calls.append((cypher, params))
            return _Result()

    graph = _Graph()
    client = RedisGraphQueryClient.__new__(RedisGraphQueryClient)
    client._graph = graph

    result = client.graph_query("RETURN $node_id", {"node_id": "sub-node-a"})

    assert result == [["payload", "identity"]]
    assert graph.calls == [("RETURN $node_id", {"node_id": "sub-node-a"})]


def test_normalize_rows_parses_raw_graph_query_header_and_stats():
    raw = [
        [["n.payload_json", 1], ["n.identity_key", 1]],
        [["{\"node_id\":\"sub-node-a\"}", "id-a"]],
        ["Cached execution: 0", "Query internal execution time: 0.1 milliseconds"],
    ]

    assert _normalize_rows(raw) == [
        {"payload_json": "{\"node_id\":\"sub-node-a\"}", "identity_key": "id-a"}
    ]
