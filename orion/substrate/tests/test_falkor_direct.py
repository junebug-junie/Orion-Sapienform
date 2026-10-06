"""FalkorDirectConceptStore: hydration-free concept-region reads.

Two lanes:

* Unit tests (always run) use a scripted fake client: query count on a turn
  with no label match, no snapshot surface, single-node read edge cases, and
  the write path.
* Equivalence tests need a real FalkorDB because the ranking and edge cut run
  in Cypher. They run when ``ORION_TEST_FALKOR_URI`` points at a throwaway
  instance (e.g. ``docker run --rm -p 127.0.0.1:16399:6379
  falkordb/falkordb``) and skip otherwise. Each test writes a seeded fixture
  graph under a unique graph name, hydrates a normal ``FalkorSubstrateStore``
  from it, and requires the direct reads to equal the cache's
  ``read_concept_region`` exactly (same models, same order). Never point
  this at the production FalkorDB: the fixture writes and then deletes its
  graph.
"""

from __future__ import annotations

import os
import random
import time
import uuid
from datetime import datetime, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    EntityNodeV1,
    EvidenceNodeV1,
    NodeRefV1,
    SubstrateActivationV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.falkor_codec import EXTERNALLY_OWNED_METADATA_KEYS
from orion.substrate.falkor_direct import (
    CONCEPT_EDGE_CUT_CYPHER,
    CONCEPT_RANK_CYPHER,
    CONCEPT_ROWS_CYPHER,
    NODE_BY_ID_CYPHER,
    FalkorDirectConceptStore,
    build_falkor_direct_concept_store_from_env,
)
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

_NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)


def _prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred", source_kind="test", source_channel="test", producer="test_falkor_direct"
    )


def _signals(salience: float, confidence: float, activation: float = 0.2) -> SubstrateSignalBundleV1:
    return SubstrateSignalBundleV1(
        salience=salience, confidence=confidence, activation=SubstrateActivationV1(activation=activation)
    )


def _concept(node_id: str, label: str, salience: float = 0.5, confidence: float = 0.5) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        label=label,
        definition=f"def of {label}",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=_NOW),
        signals=_signals(salience, confidence),
        provenance=_prov(),
    )


# ── unit lane: scripted client ──────────────────────────────────────────────


class _ScriptedClient:
    def __init__(self, responses: dict[str, list[dict]] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._responses = responses or {}
        self.read_only = True

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        return list(self._responses.get(cypher, []))


class _NoHydrateWriter(FalkorSubstrateStore):
    def _hydrate_from_durable(self) -> None:  # pragma: no cover - must never run
        raise AssertionError("writer must never hydrate")

    def snapshot(self):  # pragma: no cover - must never run
        raise AssertionError("writer must never snapshot")


def _direct(read_responses=None):
    read = _ScriptedClient(read_responses)
    write = _ScriptedClient()
    write.read_only = False
    writer = _NoHydrateWriter(FalkorSubstrateStoreConfig(uri="redis://unused"), client=write, hydrate=False)
    return FalkorDirectConceptStore(read_client=read, writer=writer), read, write


def test_no_label_match_costs_exactly_one_light_query() -> None:
    store, read, _write = _direct({CONCEPT_RANK_CYPHER: [{"object_id": 1, "label": "zebra"}]})
    region = store.read_concept_region_matching(keep_label=lambda label: False, limit_nodes=500, limit_edges=500)
    assert region.nodes == [] and region.edges == []
    assert [cypher for cypher, _ in read.calls] == [CONCEPT_RANK_CYPHER]
    assert read.calls[0][1] == {"limit_nodes": 500}


def test_rank_query_never_returns_full_rows() -> None:
    returned = CONCEPT_RANK_CYPHER.split("RETURN", 1)[1]
    assert "definition" not in returned and "payload_json" not in returned
    assert "LIMIT $limit_nodes" in CONCEPT_RANK_CYPHER
    assert "LIMIT $limit_edges" in CONCEPT_EDGE_CUT_CYPHER
    assert "id(n) = object_id" in CONCEPT_ROWS_CYPHER


def test_store_has_no_snapshot_surface() -> None:
    store, _read, _write = _direct()
    assert not hasattr(store, "snapshot")
    assert not hasattr(store, "_hydrate_from_durable")


def test_single_node_reads_go_to_falkor_and_reject_duplicates() -> None:
    row = {
        "node_id": "c1", "node_kind": "concept", "identity_key": "concept:c1", "label": "alpha",
        "anchor_scope": "orion", "observed_at": _NOW.isoformat(), "provenance_authority": "local_inferred",
        "provenance_source_kind": "test", "provenance_source_channel": "test", "provenance_producer": "t",
        "activation": 0.3,
    }
    store, read, _write = _direct({NODE_BY_ID_CYPHER: [row]})
    node = store.get_node_by_id("c1")
    assert node is not None and node.node_id == "c1" and node.signals.activation.activation == 0.3
    assert store.get_identity_key_by_node_id("c1") == "concept:c1"
    assert all(cypher == NODE_BY_ID_CYPHER for cypher, _ in read.calls)
    assert read.calls[0][1] == {"node_id": "c1"}

    dup_store, _r, _w = _direct({NODE_BY_ID_CYPHER: [row, dict(row)]})
    both, _r, _w = _direct({NODE_BY_ID_CYPHER: [row]})
    node2, identity2 = both.get_node_and_identity_key("c1")
    assert node2 == node and identity2 == "concept:c1"
    assert len(_r.calls) == 1  # one query for both halves
    assert dup_store.get_node_and_identity_key("c1") == (None, None)
    assert dup_store.get_node_by_id("c1") is None
    assert dup_store.get_identity_key_by_node_id("c1") is None
    missing, _r, _w = _direct({})
    assert missing.get_node_by_id("nope") is None


def test_upsert_is_one_merge_on_the_write_client_only() -> None:
    store, read, write = _direct()
    store.upsert_node(
        identity_key="concept:c1",
        node=_concept("c1", "alpha"),
        skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS,
    )
    assert read.calls == []
    assert len(write.calls) == 1
    cypher, params = write.calls[0]
    assert cypher.startswith("MERGE (n:SubstrateNode:")
    assert "prediction_error" not in cypher.split("SET", 1)[1]
    assert params["identity_key"] == "concept:c1"


def test_builder_uses_a_read_only_client_and_never_hydrates(monkeypatch) -> None:
    import orion.substrate.falkor_direct as mod

    built: list[dict] = []
    clients: list = []

    class _FakeRedisClient:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.read_only = bool(kwargs.get("read_only"))
            self.queries: list[str] = []
            clients.append(self)

        def graph_query(self, cypher, *_a, **_k):
            # The only thing construction may run is the node_id index DDL.
            self.queries.append(cypher)
            if not cypher.startswith("CREATE INDEX"):
                raise AssertionError("construction must not read the graph")
            raise RuntimeError("Attribute 'node_id' is already indexed")

    monkeypatch.setattr(mod, "RedisGraphQueryClient", _FakeRedisClient)
    monkeypatch.setattr(
        mod.FalkorSubstrateStore, "_hydrate_from_durable", lambda self: (_ for _ in ()).throw(AssertionError("hydrate"))
    )
    monkeypatch.setenv("FALKORDB_URI", "redis://falkor.test:6379")
    monkeypatch.setenv("FALKORDB_SUBSTRATE_GRAPH", "g1")
    store = build_falkor_direct_concept_store_from_env(socket_timeout_s=5.0, socket_connect_timeout_s=2.0)
    assert isinstance(store, FalkorDirectConceptStore)
    assert [b.get("read_only", False) for b in built] == [True, False]
    assert all(b["socket_timeout"] == 5.0 and b["graph_name"] == "g1" for b in built)
    reader, writer = clients
    assert reader.queries == []
    assert writer.queries == ["CREATE INDEX FOR (n:SubstrateNode) ON (n.node_id)"]

    monkeypatch.delenv("FALKORDB_URI")
    assert build_falkor_direct_concept_store_from_env() is None


# ── equivalence lane: real throwaway FalkorDB ───────────────────────────────

_FALKOR_URI = os.getenv("ORION_TEST_FALKOR_URI", "").strip()
live = pytest.mark.skipif(not _FALKOR_URI, reason="ORION_TEST_FALKOR_URI not set (throwaway FalkorDB)")

_PREDICATES = ["supports", "refines", "associated_with", "causes", "part_of"]
_SALIENCES = [0.0, 0.1, 0.5, 0.5, 0.5, 0.9, 1.0]  # repeated values force ties
_CONFIDENCES = [0.2, 0.5, 0.9]


def _build_fixture_graph(graph_name: str, *, seed: int, concepts: int, edges: int) -> list[str]:
    from orion.graph.falkor_client import RedisGraphQueryClient

    rng = random.Random(seed)
    client = RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name)
    writer = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=_FALKOR_URI, graph_name=graph_name), client=client, hydrate=False)
    words = ["gpu", "orion", "juniper", "memory", "camera", "fall", "sensor", "music", "garden", "self"]
    node_refs: list[NodeRefV1] = []
    labels: list[str] = []
    for i in range(concepts):
        label = f"{rng.choice(words)} {rng.choice(words)} {i}" if i % 3 else rng.choice(words)
        labels.append(label)
        node = _concept(f"c{i:03d}", label, rng.choice(_SALIENCES), rng.choice(_CONFIDENCES))
        writer.upsert_node(identity_key=f"concept:c{i:03d}", node=node)
        node_refs.append(NodeRefV1(node_id=node.node_id, node_kind="concept"))
    for i in range(concepts // 4):
        ev = EvidenceNodeV1(
            node_id=f"ev{i:03d}", evidence_type="chat_turn", content_ref=f"turn:{i}", anchor_scope="orion",
            temporal=SubstrateTemporalWindowV1(observed_at=_NOW), signals=_signals(1.0, 1.0), provenance=_prov(),
        )
        writer.upsert_node(identity_key=f"evidence:ev{i:03d}", node=ev)
        node_refs.append(NodeRefV1(node_id=ev.node_id, node_kind="evidence"))
        ent = EntityNodeV1(
            node_id=f"en{i:03d}", label=f"gpu entity {i}", anchor_scope="orion",
            temporal=SubstrateTemporalWindowV1(observed_at=_NOW), signals=_signals(1.0, 1.0), provenance=_prov(),
        )
        writer.upsert_node(identity_key=f"entity:en{i:03d}", node=ent)
        node_refs.append(NodeRefV1(node_id=ent.node_id, node_kind="entity"))
    for i in range(edges):
        source = node_refs[rng.randrange(len(node_refs))]
        target = source if i % 97 == 0 else node_refs[rng.randrange(len(node_refs))]  # a few self-loops
        edge = SubstrateEdgeV1(
            edge_id=f"e{i:05d}", source=source, target=target, predicate=rng.choice(_PREDICATES),
            temporal=SubstrateTemporalWindowV1(observed_at=_NOW), salience=rng.choice(_SALIENCES),
            confidence=rng.choice(_CONFIDENCES), provenance=_prov(),
        )
        writer.upsert_edge(identity_key=f"{source.node_id}|{edge.predicate}|{target.node_id}|{i}", edge=edge)
    # Stored values the codec decodes with defaults: NULL/0 confidence -> 0.5,
    # NULL salience -> 0.0. The ranking must order these like the decoded model.
    client.graph_query("MATCH (n:SubstrateNode) WHERE n.node_id IN ['c001', 'c004'] SET n.confidence = 0.0")
    client.graph_query("MATCH (n:SubstrateNode) WHERE n.node_id IN ['c002', 'c007'] REMOVE n.confidence")
    client.graph_query("MATCH (n:SubstrateNode) WHERE n.node_id = 'c005' REMOVE n.salience")
    client.graph_query("MATCH ()-[e]->() WHERE e.edge_id IN ['e00003', 'e00008'] SET e.confidence = 0.0")
    client.graph_query("MATCH ()-[e]->() WHERE e.edge_id = 'e00011' REMOVE e.confidence, e.salience")
    return labels


@pytest.fixture(scope="module")
def fixture_graph():
    from orion.graph.falkor_client import RedisGraphQueryClient

    graph_name = f"test_falkor_direct_{uuid.uuid4().hex[:10]}"
    labels = _build_fixture_graph(graph_name, seed=7, concepts=120, edges=1500)
    yield graph_name, labels
    client = RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name)
    client._r.execute_command("GRAPH.DELETE", graph_name)


def _pair(graph_name: str):
    from orion.graph.falkor_client import RedisGraphQueryClient

    cfg = FalkorSubstrateStoreConfig(uri=_FALKOR_URI, graph_name=graph_name)
    cache_store = FalkorSubstrateStore(cfg)
    assert cache_store.last_hydrate_ok is True
    direct = FalkorDirectConceptStore(
        read_client=RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name, read_only=True),
        writer=FalkorSubstrateStore(cfg, client=RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name), hydrate=False),
    )
    return cache_store, direct


@live
@pytest.mark.parametrize("limit_nodes,limit_edges", [(500, 500), (32, 64), (40, 200), (1, 1), (120, 1500)])
def test_full_slice_equals_hydrated_cache(fixture_graph, limit_nodes, limit_edges) -> None:
    graph_name, _labels = fixture_graph
    cache_store, direct = _pair(graph_name)
    expected = cache_store.read_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)
    actual = direct.read_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)
    assert [n.node_id for n in actual.nodes] == [n.node_id for n in expected.nodes]
    assert [e.edge_id for e in actual.edges] == [e.edge_id for e in expected.edges]
    assert actual.nodes == expected.nodes
    assert actual.edges == expected.edges


@live
@pytest.mark.parametrize("needle", ["gpu", "orion", "juniper memory", "zzz-no-match", "fall"])
def test_matching_slice_equals_collector_filter_over_cache(fixture_graph, needle) -> None:
    graph_name, _labels = fixture_graph
    cache_store, direct = _pair(graph_name)

    def keep(label: str) -> bool:
        return needle in label.lower()

    full = cache_store.read_concept_region(limit_nodes=40, limit_edges=120)
    expected_nodes = [n for n in full.nodes if keep(n.label)]
    ids = {n.node_id for n in expected_nodes}
    expected_edges = [e for e in full.edges if e.source.node_id in ids or e.target.node_id in ids]
    actual = direct.read_concept_region_matching(keep_label=keep, limit_nodes=40, limit_edges=120)
    assert actual.nodes == expected_nodes
    assert actual.edges == expected_edges


@live
def test_single_node_reads_equal_cache(fixture_graph) -> None:
    graph_name, _labels = fixture_graph
    cache_store, direct = _pair(graph_name)
    for node_id in ["c000", "c001", "c002", "c005", "ev003", "en001", "missing"]:
        assert direct.get_node_by_id(node_id) == cache_store.get_node_by_id(node_id)
        assert direct.get_identity_key_by_node_id(node_id) == cache_store.get_identity_key_by_node_id(node_id)
        assert direct.get_node_and_identity_key(node_id) == (
            cache_store.get_node_by_id(node_id),
            cache_store.get_identity_key_by_node_id(node_id),
        )


@live
def test_concept_region_latency_on_a_realistic_fixture() -> None:
    """Live-shaped graph: ~870 concepts, ~37k edges (orion_substrate 2026-10-06).
    The matching read must stay well inside the recall deadline. Bound is
    loose (CI runners vary); the measured number is printed for the report."""
    from orion.graph.falkor_client import RedisGraphQueryClient

    graph_name = f"test_falkor_direct_perf_{uuid.uuid4().hex[:8]}"
    client = RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name)
    # Bulk-load with UNWIND; going through upsert_edge one by one would take minutes.
    rng = random.Random(11)
    words = ["gpu", "orion", "juniper", "memory", "camera", "fall", "sensor", "music", "garden", "self"]
    concepts = [
        {"node_id": f"c{i:04d}", "label": f"{rng.choice(words)} topic {i}", "salience": rng.random(), "confidence": rng.random()}
        for i in range(870)
    ]
    client.graph_query(
        "UNWIND $rows AS r CREATE (:SubstrateNode:Concept {node_id: r.node_id, node_kind: 'concept', "
        "identity_key: 'concept:' + r.node_id, label: r.label, anchor_scope: 'orion', salience: r.salience, "
        "confidence: r.confidence, observed_at: '2026-10-06T00:00:00+00:00', provenance_authority: 'local_inferred', "
        "provenance_source_kind: 'test', provenance_source_channel: 'test', provenance_producer: 'perf'})",
        {"rows": concepts},
    )
    try:
        for chunk in range(37):
            edges = [
                {"s": f"c{rng.randrange(870):04d}", "t": f"c{rng.randrange(870):04d}", "id": f"e{chunk}-{j}", "sal": rng.random()}
                for j in range(1000)
            ]
            client.graph_query(
                "UNWIND $rows AS r MATCH (s:SubstrateNode {node_id: r.s}), (t:SubstrateNode {node_id: r.t}) "
                "CREATE (s)-[:supports {edge_id: r.id, predicate: 'supports', substrate_edge: true, salience: r.sal, "
                "confidence: 0.5, source_kind: 'concept', target_kind: 'concept', observed_at: '2026-10-06T00:00:00+00:00', "
                "provenance_authority: 'local_inferred', provenance_source_kind: 'test', provenance_source_channel: 'test', "
                "provenance_producer: 'perf'}]->(t)",
                {"rows": edges},
            )
        direct = FalkorDirectConceptStore(
            read_client=RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name, read_only=True),
            writer=FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=_FALKOR_URI, graph_name=graph_name), client=client, hydrate=False),
        )
        timings = {}
        for name, keep in [("no_match", lambda label: False), ("match", lambda label: "gpu topic 1" in label)]:
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                direct.read_concept_region_matching(keep_label=keep, limit_nodes=500, limit_edges=500)
                samples.append((time.perf_counter() - started) * 1000)
            timings[name] = sorted(samples)[len(samples) // 2]
        print(f"concept_region_direct_latency_ms median={timings}")
        assert timings["no_match"] < 250
        assert timings["match"] < 1500
    finally:
        client._r.execute_command("GRAPH.DELETE", graph_name)


# ── node_id index bootstrap ─────────────────────────────────────────────────


def test_ensure_indexes_is_idempotent_and_never_raises(caplog) -> None:
    from orion.substrate.falkor_store import ensure_substrate_indexes

    class _Client:
        def __init__(self, error):
            self.error = error
            self.calls = []

        def graph_query(self, cypher, params=None):
            self.calls.append(cypher)
            if self.error:
                raise self.error

    fresh = _Client(None)
    assert ensure_substrate_indexes("redis://x", "g", client=fresh) is True
    assert fresh.calls == ["CREATE INDEX FOR (n:SubstrateNode) ON (n.node_id)"]
    # 4.18 and 6.0 both say exactly this on a repeat.
    assert ensure_substrate_indexes("redis://x", "g", client=_Client(RuntimeError("Attribute 'node_id' is already indexed"))) is True
    with caplog.at_level("WARNING"):
        assert ensure_substrate_indexes("redis://x", "g", client=_Client(ConnectionError("down"))) is False
    assert "falkor_substrate_index_create_failed" in caplog.text


def test_store_bootstraps_index_only_with_its_own_client(monkeypatch) -> None:
    import orion.substrate.falkor_store as fs

    seen: list = []
    monkeypatch.setattr(fs, "ensure_substrate_indexes", lambda uri, graph, **kw: seen.append((uri, graph)) or True)
    monkeypatch.setattr(fs, "RedisGraphQueryClient", lambda **kw: _ScriptedClient())
    fs.FalkorSubstrateStore(fs.FalkorSubstrateStoreConfig(uri="redis://h:1", graph_name="g"), hydrate=False)
    assert seen == [("redis://h:1", "g")]
    fs.FalkorSubstrateStore(fs.FalkorSubstrateStoreConfig(uri="redis://h:1", graph_name="g"), client=_ScriptedClient(), hydrate=False)
    fs.FalkorSubstrateStore(
        fs.FalkorSubstrateStoreConfig(uri="redis://h:1", graph_name="g", ensure_indexes=False), hydrate=False
    )
    assert seen == [("redis://h:1", "g")]


@live
@pytest.mark.parametrize("cypher_key", ["node_by_id", "merge", "edge_merge"])
def test_node_id_queries_use_the_index(cypher_key) -> None:
    """EXPLAIN on the real engine: with the index, NODE_BY_ID and the upsert
    MERGE plan an index scan instead of a label scan."""
    from orion.graph.falkor_client import RedisGraphQueryClient
    from orion.substrate.falkor_store import ensure_substrate_indexes

    graph_name = f"test_falkor_index_{uuid.uuid4().hex[:8]}"
    client = RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name)
    client.graph_query("CREATE (:SubstrateNode:Concept {node_id: 'a', node_kind: 'concept'})")
    cypher = {
        "node_by_id": NODE_BY_ID_CYPHER,
        "merge": "MERGE (n:SubstrateNode:Concept {node_id: $node_id}) SET n.activation = 0.5",
        "edge_merge": (
            "MERGE (source:SubstrateNode {node_id: $node_id}) MERGE (target:SubstrateNode {node_id: $node_id}) "
            "MERGE (source)-[e:`supports` {edge_id: 'e1'}]->(target) SET e.salience = 1"
        ),
    }[cypher_key]
    try:
        def plan():
            return "\n".join(
                # FalkorDB 4.18 refuses EXPLAIN with unbound $params; inline them.
                str(line) for line in client._r.execute_command("GRAPH.EXPLAIN", graph_name, "CYPHER node_id='a' " + cypher)
            )

        assert "Index Scan" not in plan()
        assert ensure_substrate_indexes(_FALKOR_URI, graph_name) is True
        assert ensure_substrate_indexes(_FALKOR_URI, graph_name) is True  # repeat is fine
        assert "Index Scan" in plan(), plan()
    finally:
        client._r.execute_command("GRAPH.DELETE", graph_name)
