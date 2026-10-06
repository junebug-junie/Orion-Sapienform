"""Batched neighbor fetch == the old per-neighbor fetch (memory Stage 2 PR E).

The live algorithm is driven through the same in-memory callbacks as the
frozen #2497 version (neighborhood_oracle.py), over the hub fixture, seeded
random graphs and requests, and injected node-read faults. Every receipt
field except the read timestamps must match.
"""
from __future__ import annotations

import random

import pytest

import orion.substrate.neighborhood as module
from orion.core.schemas.cognitive_substrate import EntityNodeV1, EvidenceNodeV1, NodeRefV1, SubstrateEdgeV1
from orion.substrate.evals.neighborhood_fixture import PROVENANCE, TEMPORAL, concept, graph, hub_fixture
from orion.substrate.neighborhood import NeighborhoodRequestV1, read_memory_neighborhood
from orion.substrate.tests.neighborhood_oracle import read_neighborhood_per_neighbor

STATES = ["proposed", "provisional", "canonical", "rejected"]
SCOPES = ["world", "orion", "juniper"]
PREDICATES = ["associated_with", "co_occurs_with", "causes", "supports", "part_of"]


def receipt(result):
    return (
        [n.model_dump() for n in result.focal_nodes], [n.model_dump() for n in result.neighbor_nodes],
        [e.model_dump() for e in result.internal_edges], [e.model_dump() for e in result.boundary_edges],
        result.source_kind, result.complete_for_request, result.truncated, result.degraded, result.reason,
        result.missing_focal_node_ids, result.continuations, result.consistency,
    )


def both(monkeypatch, store, request, fault=None):
    """Run old and new algorithms through identical memory callbacks."""
    live = module.read_neighborhood
    out = []
    for algorithm in (read_neighborhood_per_neighbor, live):
        def driver(request, *, nodes, algorithm=algorithm, **kwargs):
            return algorithm(request, nodes=fault(nodes) if fault else nodes, **kwargs)
        monkeypatch.setattr(module, "read_neighborhood", driver)
        out.append(read_memory_neighborhood(store, request))
    monkeypatch.setattr(module, "read_neighborhood", live)
    return out


def random_graph(rng):
    nodes = []
    for i in range(40):
        common = dict(anchor_scope=rng.choice(SCOPES), promotion_state=rng.choice(STATES),
                      temporal=TEMPORAL, provenance=PROVENANCE)
        if i % 4 == 0:
            nodes.append(EntityNodeV1(node_id=f"ent-{i:03d}", label=f"e{i}", entity_type="machine", **common))
        elif i % 7 == 0:
            nodes.append(EvidenceNodeV1(node_id=f"evi-{i:03d}", evidence_type="reverie",
                                        content_ref=f"reverie:{i}", **common))
        else:
            nodes.append(concept(f"con-{i:03d}").model_copy(update=common))
    edges = []
    for i in range(rng.randint(30, 160)):
        source, target = rng.sample(nodes, 2)
        if rng.random() < 0.3:
            target = nodes[1] if source is not nodes[1] else nodes[2]
        edges.append(SubstrateEdgeV1(
            edge_id=f"edge-{rng.randint(0, 99999):05d}-{i}", predicate=rng.choice(PREDICATES),
            source=NodeRefV1(node_id=source.node_id, node_kind=source.node_kind),
            target=NodeRefV1(node_id=target.node_id, node_kind=target.node_kind),
            temporal=TEMPORAL, provenance=PROVENANCE))
    return graph(nodes, edges), [n.node_id for n in nodes]


def random_request(rng, ids):
    focal = rng.sample(ids, rng.choice([1, 1, 2, 3, 8, 16]))
    if rng.random() < 0.1:
        focal.append("absent-node")
    return NeighborhoodRequestV1(
        focal_node_ids=tuple(focal[:16]), direction=rng.choice(["both", "incoming", "outgoing"]),
        semantic_states=rng.choice([("provisional", "canonical"), ("proposed", "provisional", "canonical")]),
        anchor_scopes=rng.choice([tuple(SCOPES), ("orion",), ("world", "juniper")]),
        internal_edge_limit=rng.choice([0, 1, 4, 12, 256]), boundary_edge_limit=rng.choice([0, 1, 3, 8, 16, 256]),
        neighbor_node_limit=rng.choice([0, 1, 2, 8, 16, 256]))


@pytest.mark.parametrize("size", [5, 40, 1500])
@pytest.mark.parametrize("budgets", [(12, 16, 16), (4, 8, 8), (0, 0, 0), (256, 256, 256), (2, 64, 3)])
def test_hub_fixture_receipts_match(monkeypatch, size, budgets):
    store, focal = hub_fixture(size)
    request = NeighborhoodRequestV1(focal_node_ids=tuple(focal), internal_edge_limit=budgets[0],
                                    boundary_edge_limit=budgets[1], neighbor_node_limit=budgets[2])
    old, new = both(monkeypatch, store, request)
    assert receipt(new) == receipt(old)


def test_seeded_random_receipts_match(monkeypatch):
    rng = random.Random(20261006)
    compared = nonempty = truncated = degraded = 0
    for _ in range(40):
        store, ids = random_graph(rng)
        for _ in range(15):
            old, new = both(monkeypatch, store, random_request(rng, ids))
            assert receipt(new) == receipt(old)
            compared += 1
            nonempty += bool(old.boundary_edges)
            truncated += old.truncated
            degraded += old.degraded
    # Not a vacuous comparison of empty reads.
    assert compared == 600 and nonempty >= 150 and truncated >= 50 and degraded >= 30


def drop(node_id):
    return lambda nodes: (lambda ids: [n for n in nodes(ids) if n.node_id != node_id])


def duplicate(node_id):
    return lambda nodes: (lambda ids: nodes(ids) + [n for n in nodes(ids) if n.node_id == node_id])


def make_ineligible(node_id):
    def fault(nodes):
        def read(ids):
            return [n.model_copy(update={"promotion_state": "rejected"}) if n.node_id == node_id else n
                    for n in nodes(ids)]
        return read
    return fault


@pytest.mark.parametrize("fault", [drop("neighbor-0003"), duplicate("neighbor-0002"),
                                   make_ineligible("neighbor-0001"), drop("focal-0"), duplicate("focal-1")])
def test_node_read_faults_fail_the_same_way(monkeypatch, fault):
    store, focal = hub_fixture(40)
    old, new = both(monkeypatch, store, NeighborhoodRequestV1(focal_node_ids=tuple(focal)), fault)
    assert receipt(new) == receipt(old)
    assert old.degraded


def test_oracle_really_is_the_per_neighbor_algorithm(monkeypatch):
    """Guard the guard: the oracle makes one nodes() call per admitted neighbor."""
    store, focal = hub_fixture(40)
    calls = []
    record = lambda nodes: (lambda ids: calls.append(list(ids)) or nodes(ids))  # noqa: E731
    old, new = both(monkeypatch, store, NeighborhoodRequestV1(focal_node_ids=tuple(focal)), record)
    neighbors = [n.node_id for n in old.neighbor_nodes]
    # oracle: focal + one call per neighbor; live: focal + one batched call
    assert calls == [sorted(focal)] + [[n] for n in neighbors] + [sorted(focal), neighbors]
