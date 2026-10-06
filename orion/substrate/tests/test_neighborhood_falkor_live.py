"""Real-FalkorDB parity for read_neighborhood and read_evidence_handles (PR E).

Runs only when ``ORION_TEST_FALKOR_URI`` points at a THROWAWAY FalkorDB
(CI runs it against 4.18.11, production's version, and 6.0.1). Each module
writes a seeded graph under a unique name and deletes it afterwards. Never
point this at production.

The fixture graph mixes states, scopes, kinds, predicates, both provenance
shapes and edge validity windows. Every Falkor read must equal the in-memory
reference read over the same graph: same nodes, same edges in the same order,
same receipts. The plan test pins the index seek that the batching patch
depends on for its latency.
"""
from __future__ import annotations

import os
import random
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1, EntityNodeV1, EvidenceNodeV1, NodeRefV1, SubstrateEdgeV1,
    SubstrateProvenanceV1, SubstrateTemporalWindowV1,
)
from orion.substrate.evidence_handles import EvidenceHandleRequestV1
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
from orion.substrate.neighborhood import NeighborhoodRequestV1
from orion.substrate.store import InMemorySubstrateGraphStore

URI = os.getenv("ORION_TEST_FALKOR_URI", "").strip()
pytestmark = pytest.mark.skipif(not URI, reason="ORION_TEST_FALKOR_URI not set (throwaway FalkorDB)")

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
PROV = SubstrateProvenanceV1(authority="local_inferred", source_kind="test", source_channel="test",
                             producer="test_neighborhood_falkor_live")
STATES = ["proposed", "provisional", "canonical", "rejected"]
SCOPES = ["world", "orion", "juniper"]
PREDICATES = ["associated_with", "co_occurs_with", "causes", "supports", "part_of"]
EVIDENCE_TYPES = ["episode_memory", "reverie", "chat_turn"]


def _when(rng):
    return NOW - timedelta(days=rng.randint(0, 20), seconds=rng.randint(0, 86399), microseconds=rng.choice([0, 1500]))


def _build(rng):
    nodes, edges = [], []
    for i in range(60):
        common = dict(anchor_scope=rng.choice(SCOPES), promotion_state=rng.choice(STATES),
                      temporal=SubstrateTemporalWindowV1(observed_at=_when(rng)), provenance=PROV)
        if i % 3:
            nodes.append(ConceptNodeV1(node_id=f"con-{i:03d}", label=f"c{i}", **common))
        else:
            nodes.append(EntityNodeV1(node_id=f"ent-{i:03d}", label=f"e{i}", entity_type="machine", **common))
    hub = nodes[1]  # a hub with many incoming edges and both directions
    for i in range(30):
        nodes.append(EvidenceNodeV1(node_id=f"evi-{i:03d}", anchor_scope="juniper",
            evidence_type=rng.choice(EVIDENCE_TYPES), content_ref=f"episode_memory:{i}",
            temporal=SubstrateTemporalWindowV1(observed_at=_when(rng)), provenance=PROV))
    semantic = [n for n in nodes if n.node_kind != "evidence"]
    evidence = [n for n in nodes if n.node_kind == "evidence"]

    def ref(node):
        return NodeRefV1(node_id=node.node_id, node_kind=node.node_kind)

    def add(source, target, predicate, **temporal):
        temporal.setdefault("observed_at", _when(rng))
        edges.append(SubstrateEdgeV1(edge_id=f"edge-{len(edges):05d}", source=ref(source), target=ref(target),
            predicate=predicate, temporal=SubstrateTemporalWindowV1(**temporal), provenance=PROV))

    for _ in range(500):
        source, target = rng.sample(semantic, 2)
        if rng.random() < 0.3:
            target = hub
            if source is hub:
                continue
        add(source, target, rng.choice(PREDICATES))
    for _ in range(200):
        node, ev = rng.choice(semantic[:12]), rng.choice(evidence)
        window = {}
        roll = rng.random()
        if roll < 0.4:
            window["valid_from"] = _when(rng)
        if roll > 0.8:
            window["valid_to"] = NOW - timedelta(days=rng.randint(-3, 10))
        if rng.random() < 0.6:
            add(node, ev, "observed_in", **window)
        else:
            add(ev, node, rng.choice(["supports", "observed_in"]), **window)
    return nodes, edges


@pytest.fixture(scope="module")
def stores():
    from orion.graph.falkor_client import RedisGraphQueryClient

    name = f"test_nbhd_live_{uuid.uuid4().hex[:10]}"
    nodes, edges = _build(random.Random(2497))
    writer_client = RedisGraphQueryClient(uri=URI, graph_name=name)
    writer = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=URI, graph_name=name),
                                  client=writer_client, hydrate=False)
    memory = InMemorySubstrateGraphStore()
    for node in nodes:
        writer.upsert_node(identity_key=node.node_id, node=node)
        memory.upsert_node(identity_key=node.node_id, node=node)
    for item in edges:
        writer.upsert_edge(identity_key=item.edge_id, edge=item)
        memory.upsert_edge(identity_key=item.edge_id, edge=item)
    try:
        writer_client.graph_query("CREATE INDEX FOR (n:SubstrateNode) ON (n.node_id)")
    except Exception:  # noqa: BLE001 - "already indexed" on a repeat create
        pass
    reader = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=URI, graph_name=name),
        client=RedisGraphQueryClient(uri=URI, graph_name=name, read_only=True), hydrate=False)
    reader.snapshot = lambda: pytest.fail("hydration forbidden")
    yield memory, reader, writer_client, nodes
    writer_client._r.execute_command("GRAPH.DELETE", name)


def _neighborhood_receipt(result):
    return ([n.node_id for n in result.focal_nodes], [n.node_id for n in result.neighbor_nodes],
            [e.edge_id for e in result.internal_edges], [e.edge_id for e in result.boundary_edges],
            result.complete_for_request, result.truncated, result.degraded, result.reason,
            result.missing_focal_node_ids)


def test_neighborhood_parity_over_random_requests(stores):
    memory, reader, _, nodes = stores
    semantic = [n.node_id for n in nodes if n.node_kind != "evidence"]
    rng = random.Random(6)
    checked = nonempty = 0
    for _ in range(80):
        focal = rng.sample(semantic, rng.choice([1, 1, 2, 4, 16])) + (["absent-id"] if rng.random() < 0.1 else [])
        request = NeighborhoodRequestV1(
            focal_node_ids=focal[:16], direction=rng.choice(["both", "incoming", "outgoing"]),
            semantic_states=rng.choice([("provisional", "canonical"), ("proposed", "provisional", "canonical")]),
            anchor_scopes=rng.choice([("world", "orion", "juniper"), ("orion",)]),
            internal_edge_limit=rng.choice([0, 2, 12]), boundary_edge_limit=rng.choice([0, 3, 8, 64]),
            neighbor_node_limit=rng.choice([0, 1, 8, 64]))
        expected = memory.read_neighborhood(request)
        actual = reader.read_neighborhood(request)
        assert actual.source_kind == "falkor"
        assert _neighborhood_receipt(actual) == _neighborhood_receipt(expected), request
        checked += 1
        nonempty += bool(actual.boundary_edges)
    assert checked == 80 and nonempty >= 20  # not a vacuous all-empty comparison


def test_evidence_handle_parity_over_random_requests(stores):
    memory, reader, _, nodes = stores
    semantic = [n.node_id for n in nodes if n.node_kind != "evidence"]
    rng = random.Random(7)
    nonempty = truncated = 0
    for _ in range(80):
        ids = rng.sample(semantic[:12], rng.choice([1, 2, 5])) + rng.sample(semantic, 1)
        if rng.random() < 0.15:
            ids.append("evi-000")  # an evidence node is not a semantic node
        request = EvidenceHandleRequestV1(
            node_ids=tuple(ids), per_node_limit=rng.choice([1, 3, 6, 16]),
            evidence_types=rng.choice([None, ("episode_memory",), ("reverie", "chat_turn")]),
            at=NOW - timedelta(days=rng.choice([0, 0, 2, 8, 30])))
        expected = memory.read_evidence_handles(request)
        actual = reader.read_evidence_handles(request)
        assert actual.source_kind == "falkor"
        assert actual.handles == expected.handles, request
        assert (actual.missing_node_ids, actual.truncated_node_ids, actual.degraded, actual.reason) == (
            expected.missing_node_ids, expected.truncated_node_ids, expected.degraded, expected.reason)
        nonempty += bool(actual.handles)
        truncated += actual.truncated
    assert nonempty >= 40 and truncated >= 5


def _cypher_params(params):
    def lit(value):
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(lit(v) for v in value) + "]"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"
    return "CYPHER " + " ".join(f"{key}={lit(value)}" for key, value in params.items()) + " "


def test_every_issued_read_seeks_the_node_id_index(stores):
    """The latency win depends on these plans: no read may label-scan.

    Records the exact queries the reader issues (hub focal, both directions,
    plus an evidence-handle read) and EXPLAINs each with its parameters.
    """
    _, reader, client, nodes = stores
    issued = []
    inner = reader._client

    class Recording:
        def graph_query(self, query, params=None):
            issued.append((query, dict(params or {})))
            return inner.graph_query(query, params=params)

    reader._client = Recording()
    try:
        hub = nodes[1].node_id
        result = reader.read_neighborhood(NeighborhoodRequestV1(
            focal_node_ids=(hub,), semantic_states=tuple(STATES), anchor_scopes=tuple(SCOPES)))
        assert result.boundary_edges and not result.degraded
        assert reader.read_evidence_handles(EvidenceHandleRequestV1(node_ids=(hub,), at=NOW)).source_kind == "falkor"
    finally:
        reader._client = inner
    kinds = set()
    for query, params in issued:
        plan = client._r.execute_command("GRAPH.EXPLAIN", client._graph_name, _cypher_params(params) + query)
        text = "\n".join(p.decode() if isinstance(p, bytes) else str(p) for p in plan)
        assert "Label Scan" not in text and "All Node Scan" not in text, (query, text)
        kinds.add("incoming" if "WITH target" in query else "outgoing" if "WITH source" in query else
                  "nodes" if "MATCH (n:SubstrateNode)" in query and "OPTIONAL" not in query else
                  "evidence" if "OPTIONAL" in query else "internal")
    assert {"incoming", "outgoing", "nodes", "evidence", "internal"} <= kinds, kinds
