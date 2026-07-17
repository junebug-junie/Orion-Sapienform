from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    DriveNodeV1,
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
        taxonomy_path=["root"],
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_falkor_store",
            evidence_refs=["ev:store"],
        ),
        metadata=metadata or {},
    )


def _assert_no_payload_json_sor(cypher: str, params: dict | None) -> None:
    assert "$payload_json" not in cypher
    assert "payload_json =" not in cypher.replace("REMOVE n.payload_json", "").replace(
        "REMOVE e.payload_json", ""
    )
    assert params is not None
    assert "payload_json" not in params


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
    _assert_no_payload_json_sor(cypher, params)
    assert "REMOVE n.payload_json" in cypher
    assert "MERGE (n:SubstrateNode:Concept {node_id: $node_id})" in cypher
    assert "n.label = $label" in cypher
    assert "n.promotion_state = $promotion_state" in cypher
    assert "n.salience = $salience" in cypher
    assert "n.evidence_refs_json = $evidence_refs_json" in cypher
    assert "n.taxonomy_path_json = $taxonomy_path_json" in cypher
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
    assert params["evidence_refs_json"] == '["ev:store"]'
    assert params["taxonomy_path_json"] == '["root"]'


def test_falkor_rejects_non_concept_durable_write():
    client = RecordingFalkorClient()
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=False,
    )
    drive = DriveNodeV1(
        node_id="drive-curiosity",
        drive_kind="curiosity",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_falkor_store",
        ),
    )

    with pytest.raises(ValueError, match="concept nodes only"):
        store.upsert_node(identity_key="drive:curiosity", node=drive)

    assert client.calls == []


def test_falkor_hydrates_concept_from_native_properties():
    client = RecordingFalkorClient(
        hydrate_node_rows=[
            {
                "node_id": "concept-hydrated",
                "node_kind": "concept",
                "identity_key": "concept:hydrated",
                "label": "Hydrated concept",
                "definition": "Loaded from Falkor native properties",
                "taxonomy_path_json": '["atlas"]',
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
                "evidence_refs_json": '["ev:hydrate"]',
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
    assert node.taxonomy_path == ["atlas"]
    assert node.promotion_state == "canonical"
    assert node.signals.confidence == 0.75
    assert node.signals.salience == 0.66
    assert node.signals.activation.activation == 0.44
    assert node.signals.activation.recency_score == 0.33
    assert node.provenance.evidence_refs == ["ev:hydrate"]
    assert store.get_node_id_by_identity("concept:hydrated") == "concept-hydrated"


def test_falkor_hydrates_legacy_payload_json_and_rewrites_native():
    legacy_node = ConceptNodeV1(
        node_id="concept-legacy",
        label="Legacy concept",
        definition="From blob SoR",
        taxonomy_path=["legacy"],
        anchor_scope="orion",
        promotion_state="canonical",
        temporal=SubstrateTemporalWindowV1(
            observed_at=datetime(2026, 7, 16, tzinfo=timezone.utc)
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test:legacy",
            producer="test_falkor_store",
            evidence_refs=["ev:legacy"],
        ),
    )
    client = RecordingFalkorClient(
        hydrate_legacy_node_rows=[
            {
                "payload_json": legacy_node.model_dump_json(),
                "identity_key": "concept:legacy",
            }
        ]
    )

    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    node = store.get_node_by_id("concept-legacy")
    assert node is not None
    assert node.label == "Legacy concept"
    assert node.taxonomy_path == ["legacy"]
    assert node.provenance.evidence_refs == ["ev:legacy"]
    assert store.get_node_id_by_identity("concept:legacy") == "concept-legacy"

    write_calls = [(c, p) for c, p in client.calls if "MERGE (n:SubstrateNode" in c]
    assert write_calls, "expected native rewrite after legacy hydrate"
    cypher, params = write_calls[-1]
    _assert_no_payload_json_sor(cypher, params)
    assert "REMOVE n.payload_json" in cypher
    assert params["label"] == "Legacy concept"
    assert params["evidence_refs_json"] == '["ev:legacy"]'


def test_recording_client_splits_node_and_edge_hydrate_rows():
    client = RecordingFalkorClient(
        hydrate_node_rows=[{"node_id": "n1", "node_kind": "concept"}],
        hydrate_edge_rows=[{"edge_id": "e1", "predicate": "supports"}],
    )

    node_rows = client.graph_query(
        "MATCH (n:SubstrateNode) RETURN n.node_id AS node_id, n.node_kind AS node_kind"
    )
    edge_rows = client.graph_query(
        "MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode) "
        "WHERE e.substrate_edge = true "
        "RETURN e.edge_id AS edge_id"
    )

    assert node_rows == [{"node_id": "n1", "node_kind": "concept"}]
    assert edge_rows == [{"edge_id": "e1", "predicate": "supports"}]


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
    _assert_no_payload_json_sor(cypher, params)
    assert "REMOVE e.payload_json" in cypher
    assert "e.substrate_edge = $substrate_edge" in cypher
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
        [["n.node_id", 1], ["n.identity_key", 1]],
        [["concept-alpha", "concept:alpha"]],
        ["Cached execution: 0", "Query internal execution time: 0.1 milliseconds"],
    ]

    assert _normalize_rows(raw) == [
        {"node_id": "concept-alpha", "identity_key": "concept:alpha"}
    ]


def test_falkor_hydrated_concepts_support_concept_region_query():
    client = RecordingFalkorClient(
        hydrate_node_rows=[
            {
                "node_id": "concept-alpha",
                "node_kind": "concept",
                "identity_key": "concept:alpha",
                "label": "Alpha",
                "definition": None,
                "taxonomy_path_json": "[]",
                "anchor_scope": "orion",
                "subject_ref": None,
                "promotion_state": "canonical",
                "risk_tier": "low",
                "confidence": 0.8,
                "salience": 0.7,
                "activation": 0.5,
                "recency_score": 0.4,
                "decay_floor": 0.0,
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
                "evidence_refs_json": "[]",
            }
        ]
    )
    store = FalkorSubstrateStore(
        FalkorSubstrateStoreConfig(uri="redis://localhost:6379", graph_name="orion_substrate"),
        client=client,
        hydrate=True,
    )

    result = store.query_concept_region(limit_nodes=10, limit_edges=10)

    assert result.source_kind == "falkor"
    assert [node.node_id for node in result.slice.nodes] == ["concept-alpha"]
