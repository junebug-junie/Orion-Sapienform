from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1, EvidenceNodeV1, NodeRefV1, SubstrateEdgeV1,
    SubstrateProvenanceV1, SubstrateTemporalWindowV1,
)
from orion.substrate.neighborhood import NeighborhoodRequestV1, read_neighborhood
from orion.substrate.store import InMemorySubstrateGraphStore
from orion.substrate.routed_store import RoutedSubstrateGraphStore

from orion.substrate.evals.neighborhood_fixture import concept, edge, graph, hub_fixture, PROVENANCE, TEMPORAL

def assert_endpoints(result):
    nodes = {n.node_id: n for n in result.focal_nodes + result.neighbor_nodes}
    for item in result.internal_edges + result.boundary_edges:
        for ref in (item.source, item.target):
            assert nodes[ref.node_id].node_kind == ref.node_kind


def test_internal_budget_survives_high_ranked_boundary_and_round_robin():
    store, focal = hub_fixture()
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=focal))
    assert len(result.internal_edges) == 4
    assert len(result.boundary_edges) == 16
    assert {"small-in", "small-out"} <= {e.edge_id for e in result.boundary_edges}
    assert result.truncated and not result.complete_for_request and not result.degraded
    assert_endpoints(result)


@pytest.mark.parametrize("direction, expected", [("incoming", {"incoming"}), ("outgoing", {"outgoing"}), ("both", {"incoming", "outgoing"})])
def test_direction_and_complete_receipt(direction, expected):
    store = graph([concept("aaa"), concept("bbb"), concept("ccc")],
        [edge("incoming", "bbb", "aaa"), edge("outgoing", "aaa", "ccc")])
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=["aaa"], direction=direction))
    assert {e.edge_id for e in result.boundary_edges} == expected
    assert result.complete_for_request and not result.truncated
    assert result.read_finished_at >= result.read_started_at
    assert_endpoints(result)


def test_neighbor_limit_and_zero_budgets_are_honest():
    store, focal = hub_fixture(5)
    for neighbor_limit in (0, 1, 2):
        result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=focal, neighbor_node_limit=neighbor_limit))
        assert len(result.neighbor_nodes) <= neighbor_limit
        assert result.truncated
        assert_endpoints(result)
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=focal, internal_edge_limit=0, boundary_edge_limit=0))
    assert not result.internal_edges and not result.boundary_edges and result.truncated


def test_scope_state_and_provenance_nodes_do_not_compete():
    evidence = EvidenceNodeV1(node_id="evidence", anchor_scope="world", promotion_state="canonical",
        evidence_type="reading", content_ref="artifact:1", temporal=TEMPORAL, provenance=PROVENANCE)
    store = graph([concept("aaa"), concept("bbb").model_copy(update={"promotion_state": "proposed"}),
        concept("ccc").model_copy(update={"anchor_scope": "juniper"}), evidence],
        [edge("proposed", "aaa", "bbb"), edge("private", "aaa", "ccc"),
         edge("provenance", "aaa", "evidence").model_copy(update={"target": NodeRefV1(node_id="evidence", node_kind="evidence")})])
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=["aaa"], anchor_scopes=["world"]))
    assert result.complete_for_request and not result.boundary_edges


def test_missing_focal_not_empty_success_and_unsupported_cursor_requires_restart():
    store = graph([], [])
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=["absent"]))
    assert result.degraded and not result.complete_for_request
    assert result.missing_focal_node_ids == ("absent",)
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=[], continuation="old"))
    assert result.degraded and result.reason.startswith("restart_required")
    assert store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=[])).complete_for_request


def test_routed_only_reads_primary_without_snapshot():
    store = graph([concept("aaa")], [])
    store.snapshot = lambda: pytest.fail("hydration forbidden")
    class Shadow:
        def read_neighborhood(self, request):
            pytest.fail("shadow read forbidden")
    result = RoutedSubstrateGraphStore(primary=store, shadow=Shadow()).read_neighborhood(NeighborhoodRequestV1(focal_node_ids=["aaa"]))
    assert result.complete_for_request


def test_short_pages_continue_and_failure_never_returns_partial_success():
    nodes = [concept("aaa"), concept("bbb")]
    items = [edge(f"edge-{i}", "aaa", "bbb") for i in range(4)]
    calls = []
    def fetch(ids, group, after, limit):
        calls.append(after)
        return [e for e in items if e.edge_id > after][:1]
    result = read_neighborhood(NeighborhoodRequestV1(focal_node_ids=["aaa", "bbb"]),
        source_kind="test", nodes=lambda ids: nodes, groups=lambda ids: [], edges=fetch)
    assert len(result.internal_edges) == 4 and result.complete_for_request
    assert len(calls) == 5
    def fail(ids, group, after, limit):
        if after:
            raise RuntimeError("secret endpoint credentials")
        return items[:1]
    result = read_neighborhood(NeighborhoodRequestV1(focal_node_ids=["aaa", "bbb"]),
        source_kind="test", nodes=lambda ids: nodes, groups=lambda ids: [], edges=fail)
    assert result.degraded and not result.internal_edges
    assert "secret" not in result.reason


def test_duplicate_ids_and_changed_endpoint_fail_closed():
    aaa = concept("aaa")
    for nodes, edges, groups in [
        (lambda ids: [aaa, aaa], lambda *args: [], lambda ids: []),
        (lambda ids: [aaa], lambda *args: [edge("same", "aaa", "aaa")] * 2, lambda ids: []),
        (lambda ids: [aaa] if ids == ["aaa"] else [],
         lambda ids, group, after, limit: [edge("bbb-edge", "aaa", "bbb")] if group and not after else [],
         lambda ids: [("aaa", "outgoing", "associated_with")]),
    ]:
        result = read_neighborhood(NeighborhoodRequestV1(focal_node_ids=["aaa"]),
            source_kind="test", nodes=nodes, edges=edges, groups=groups)
        assert result.degraded and not result.complete_for_request


@pytest.mark.parametrize("kwargs", [{"internal_edge_limit": -1}, {"boundary_edge_limit": 257}, {"direction": "sideways"}, {"focal_node_ids": ["aaa"] * 17}])
def test_request_bounds(kwargs):
    with pytest.raises(ValidationError):
        NeighborhoodRequestV1(**({"focal_node_ids": ["aaa"]} | kwargs))


def test_focal_fairness_precedes_direction_and_predicate_diversity():
    store = graph([concept(key) for key in ("aaa", "bbb", "xxx", "yyy", "zzz")],
        [edge("hub-in", "xxx", "aaa"), edge("hub-out", "aaa", "yyy"),
         edge("small-in", "zzz", "bbb")])
    result = store.read_neighborhood(NeighborhoodRequestV1(
        focal_node_ids=["aaa", "bbb"], boundary_edge_limit=2))
    assert {e.edge_id for e in result.boundary_edges} == {"hub-in", "small-in"}
    assert result.truncated
