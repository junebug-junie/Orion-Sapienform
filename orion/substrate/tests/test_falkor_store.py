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
