"""Proposed endpoints of an accepted semantic projection (2026-10-10).

Almost every live concept is `proposed`. A reading claim (#2581) accepted as
`provisional` projects a semantic_projection edge between two such concepts. A
default neighborhood read must return that edge with both endpoints; the edge's
own acceptance (walkable_edge) is what authorizes it. Nothing else about
proposed nodes changes: no legacy edge, no provenance edge, no rejected claim.
"""
from __future__ import annotations

import pytest
import rdflib

from orion.core.schemas.cognitive_substrate import AssertionNodeV1, NodeRefV1
from orion.substrate.evals.neighborhood_fixture import PROVENANCE, TEMPORAL, concept, edge, graph
from orion.substrate.graphdb_store import GraphDBSubstrateStore, GraphDBSubstrateStoreConfig
from orion.substrate.neighborhood import NeighborhoodRequestV1, read_neighborhood
from orion.substrate.query_planning import (
    SubstrateQueryPlanStepV1, SubstrateQueryPlanV1, SubstrateSemanticReadCoordinator,
)


def proposed(node_id, **kwargs):
    return concept(node_id, **kwargs).model_copy(update={"promotion_state": "proposed"})


def claim(node_id="claim-1", *, state="provisional", revision=1):
    return AssertionNodeV1(node_id=node_id, anchor_scope="world", promotion_state=state, temporal=TEMPORAL,
                           provenance=PROVENANCE, predicate="associated_with", statement_key=f"k|{node_id}",
                           statement_text="a quoted reading claim", revision=revision)


def projection(edge_id="proj", src="read-a", dst="read-b", *, claim_id="claim-1", revision=1):
    return edge(edge_id, src, dst, edge_role="semantic_projection", assertion_id=claim_id,
                assertion_revision=revision)


def reading_store(*, state="provisional", revision=1, edge_revision=1, extra_nodes=(), extra_edges=()):
    return graph([proposed("read-a"), proposed("read-b"), claim(state=state, revision=revision), *extra_nodes],
                 [projection(revision=edge_revision), *extra_edges])


def receipt(result):
    return (sorted(n.node_id for n in result.focal_nodes), sorted(n.node_id for n in result.neighbor_nodes),
            sorted(e.edge_id for e in result.internal_edges), sorted(e.edge_id for e in result.boundary_edges))


@pytest.mark.parametrize("focal", ["read-a", "read-b"])
def test_default_read_returns_an_accepted_link_between_two_proposed_concepts(focal):
    result = reading_store().read_neighborhood(NeighborhoodRequestV1(focal_node_ids=(focal,)))
    other = "read-b" if focal == "read-a" else "read-a"
    assert not result.degraded, result.reason
    assert receipt(result) == ([focal], [other], [], ["proj"])
    assert result.complete_for_request and not result.missing_focal_node_ids
    # Both endpoints are marked as admitted by the projection, not by their own state.
    assert result.projection_endpoint_node_ids == ("read-a", "read-b")


def test_both_proposed_focals_read_the_link_as_internal():
    result = reading_store().read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a", "read-b")))
    assert receipt(result) == (["read-a", "read-b"], [], ["proj"], [])
    assert result.complete_for_request


def test_canonical_focal_with_proposed_neighbor_marks_only_the_neighbor():
    store = graph([concept("seed"), proposed("read-b"), claim()], [projection(src="seed")])
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("seed",)))
    assert receipt(result) == (["seed"], ["read-b"], [], ["proj"])
    assert result.projection_endpoint_node_ids == ("read-b",)


@pytest.mark.parametrize("state,revision,edge_revision", [
    ("rejected", 1, 1), ("deprecated", 1, 1), ("proposed", 1, 1),  # never or no longer accepted
    ("provisional", 2, 1),                                          # stale projection
])
def test_an_unaccepted_or_stale_projection_admits_nothing(state, revision, edge_revision):
    store = reading_store(state=state, revision=revision, edge_revision=edge_revision)
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a",)))
    assert receipt(result) == ([], [], [], [])
    assert result.missing_focal_node_ids == ("read-a",)
    assert result.reason == "focal_unavailable_or_filtered"


def test_a_missing_claim_admits_nothing():
    store = graph([proposed("read-a"), proposed("read-b")], [projection(claim_id="gone")])
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a",)))
    assert result.missing_focal_node_ids == ("read-a",)


def test_proposed_node_with_only_legacy_and_provenance_edges_stays_excluded():
    from orion.core.schemas.cognitive_substrate import EvidenceNodeV1
    evidence = EvidenceNodeV1(node_id="evidence", anchor_scope="world", promotion_state="canonical",
        evidence_type="reading", content_ref="artifact:1", temporal=TEMPORAL, provenance=PROVENANCE)
    store = graph([proposed("lonely"), concept("seed"), proposed("other"), evidence],
        [edge("legacy-out", "lonely", "seed"), edge("legacy-in", "other", "lonely"),
         edge("prov", "lonely", "evidence", edge_role="provenance").model_copy(
             update={"target": NodeRefV1(node_id="evidence", node_kind="evidence")})])
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("lonely",)))
    assert receipt(result) == ([], [], [], [])
    assert result.missing_focal_node_ids == ("lonely",)
    # ...and a canonical focal still does not walk a legacy edge into a proposed node.
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("seed",)))
    assert receipt(result) == (["seed"], [], [], [])


def test_a_proposed_endpoint_walks_its_projection_but_not_its_legacy_edges():
    store = reading_store(extra_nodes=[concept("seed"), proposed("junk")],
                          extra_edges=[edge("legacy-seed", "read-a", "seed"), edge("legacy-junk", "junk", "read-a")])
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a",)))
    assert receipt(result) == (["read-a"], ["read-b"], [], ["proj"])


def test_rejected_endpoint_is_not_admitted_even_by_an_accepted_projection():
    store = graph([proposed("read-a"), proposed("read-b").model_copy(update={"promotion_state": "rejected"}),
                   claim()], [projection()])
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a",)))
    assert receipt(result) == ([], [], [], [])


def test_direction_gates_a_proposed_focal():
    store = reading_store()  # read-a -> read-b
    assert receipt(store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a",), direction="incoming"))) \
        == ([], [], [], [])
    assert receipt(store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-b",), direction="incoming"))) \
        == (["read-b"], ["read-a"], [], ["proj"])


def test_opting_out_restores_the_node_state_only_rule():
    request = NeighborhoodRequestV1(focal_node_ids=("read-a",), projection_endpoint_states=())
    result = reading_store().read_neighborhood(request)
    assert result.missing_focal_node_ids == ("read-a",) and not result.boundary_edges


def test_budgets_and_truncation_still_bind_projection_neighbors():
    claims = [claim(f"claim-{i}") for i in range(6)]
    neighbors = [proposed(f"nb-{i}") for i in range(6)]
    edges = [projection(f"proj-{i}", "read-a", f"nb-{i}", claim_id=f"claim-{i}") for i in range(6)]
    store = graph([proposed("read-a"), *neighbors, *claims], edges)
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a",), neighbor_node_limit=2))
    assert len(result.neighbor_nodes) == 2 and len(result.boundary_edges) == 2
    assert result.truncated and not result.complete_for_request and result.reason == "budget_exhausted"
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a",), boundary_edge_limit=3))
    assert len(result.boundary_edges) == 3 and result.truncated
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a",), boundary_edge_limit=0))
    assert not result.boundary_edges and result.truncated


def test_a_backend_returning_a_legacy_edge_into_a_proposed_node_fails_closed():
    """No dangling edge and no smuggled endpoint: the driver re-checks the per-edge rule."""
    nodes = {"seed": concept("seed"), "junk": proposed("junk")}
    result = read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("seed",)), source_kind="test",
        nodes=lambda ids: [nodes[i] for i in ids if i in nodes],
        groups=lambda ids: [("seed", "outgoing", "associated_with")],
        edges=lambda ids, group, after, limit: [edge("legacy", "seed", "junk")] if group and not after else [])
    assert result.degraded and result.reason == "unavailable:ValueError"
    assert not result.boundary_edges and not result.neighbor_nodes


def test_planner_exposes_projection_endpoint_refs():
    execution = SubstrateSemanticReadCoordinator(store=reading_store()).execute(SubstrateQueryPlanV1(
        plan_kind="diagnostic", steps=(SubstrateQueryPlanStepV1("neighborhood", {"focal_node_ids": ["read-a"]}),)))
    details = execution.results[0].details
    assert details["boundary_edge_refs"] == ["proj"]
    assert details["projection_endpoint_node_refs"] == ["read-a", "read-b"]


def test_sparql_backend_stays_fail_closed_for_projection_endpoints():
    """GraphDB stores no Assertion nodes, so it cannot verify a projection: it keeps
    walking legacy edges only and a proposed focal stays filtered (documented choice)."""
    memory = reading_store(extra_nodes=[concept("seed"), concept("seed-2")],
                           extra_edges=[edge("legacy", "seed", "seed-2")])
    dataset = rdflib.Dataset()
    store = GraphDBSubstrateStore(GraphDBSubstrateStoreConfig(endpoint="http://unused"))
    store._update = dataset.update
    for node in memory._nodes.values():
        if node.node_kind != "assertion":
            store.upsert_node(identity_key=node.node_id, node=node)
    for item in memory._edges.values():
        store.upsert_edge(identity_key=item.edge_id, edge=item)
    store._select = lambda query: [{str(k): {"value": str(v)} for k, v in row.asdict().items()}
                                   for row in dataset.query(query)]
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("read-a",)))
    assert receipt(result) == ([], [], [], []) and result.missing_focal_node_ids == ("read-a",)
    legacy = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("seed",)))
    assert receipt(legacy) == (["seed"], ["seed-2"], [], ["legacy"])  # legacy reads unchanged
