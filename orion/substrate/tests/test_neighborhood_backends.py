import pytest
import rdflib

from orion.substrate.falkor_codec import encode_node_properties, encode_edge_properties
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
from orion.substrate.graphdb_store import GraphDBSubstrateStore, GraphDBSubstrateStoreConfig
from orion.substrate.neighborhood import NeighborhoodRequestV1
from orion.substrate.query_planning import SubstrateSemanticReadCoordinator, SubstrateQueryPlanStepV1, SubstrateQueryPlanV1
from orion.substrate.tests.test_neighborhood import concept, edge, graph, assert_endpoints
from orion.substrate.evals.neighborhood_fixture import hub_fixture


class NativeFixture:
    """Native codec rows; cap pages at one to exercise server cap handling."""
    def __init__(self, store):
        self.store = store
        self.calls = []

    def graph_query(self, query, params):
        self.calls.append((query, params))
        assert not any(word in query for word in ("MERGE", "SET ", "DELETE", "CREATE"))
        if "MATCH (n:SubstrateNode)" in query:
            # Batched: one IN-list read per call, LIMIT 2n keeps duplicates visible.
            assert "n.node_id IN $node_ids" in query
            assert f"LIMIT {2 * len(params['node_ids'])}" in query
            return [encode_node_properties(self.store._nodes[key], key)
                    for key in params["node_ids"] if key in self.store._nodes]
        assert "source.promotion_state IN $states" in query
        assert "target.anchor_scope IN $scopes" in query
        items = []
        for item in self.store._edges.values():
            src = self.store._nodes[item.source.node_id]
            dst = self.store._nodes[item.target.node_id]
            if any(n.promotion_state not in params["states"] or n.anchor_scope not in params["scopes"] or n.node_kind not in {"concept", "entity"} for n in (src, dst)):
                continue
            if "focal" in params:
                incoming = "target.node_id = $focal" in query
                # Group reads bind the focal endpoint before the expand so
                # FalkorDB seeks the node_id index instead of a label scan.
                inside = "target" if incoming else "source"
                assert query.startswith(f"MATCH ({inside}:SubstrateNode) WHERE {inside}.node_id = $focal WITH {inside} ")
                inside, outside = (dst, src) if incoming else (src, dst)
                if inside.node_id != params["focal"] or outside.node_id in params["ids"]:
                    continue
                if "predicate" in params and item.predicate != params["predicate"]:
                    continue
            elif src.node_id not in params["ids"] or dst.node_id not in params["ids"]:
                continue
            items.append(item)
        if "RETURN DISTINCT" in query:
            return [{"predicate": p} for p in sorted({e.predicate for e in items}) if p > params["after"]][:1]
        assert "source.node_id AS source_id" in query and "target.node_id AS target_id" in query
        assert "ORDER BY e.edge_id LIMIT" in query
        return [encode_edge_properties(e, e.edge_id) for e in sorted(items, key=lambda e:e.edge_id) if e.edge_id > params["after"]][:1]


def fixture_graph():
    return graph([concept("aaa"), concept("bbb"), concept("ccc"), concept("ddd"),
                  concept("proposed").model_copy(update={"promotion_state": "proposed"})],
        [edge("internal-1", "aaa", "bbb"), edge("internal-2", "bbb", "aaa"),
         edge("incoming", "ccc", "aaa"), edge("outgoing", "bbb", "ddd", "causes"),
         edge("excluded", "aaa", "proposed")])


def ids(result):
    return tuple(sorted(x.node_id for x in result.focal_nodes)), tuple(sorted(x.node_id for x in result.neighbor_nodes)), tuple(sorted(x.edge_id for x in result.internal_edges)), tuple(sorted(x.edge_id for x in result.boundary_edges))


@pytest.mark.parametrize("backend", ["falkor", "sparql"])
@pytest.mark.parametrize("direction", ["both", "incoming", "outgoing"])
def test_durable_parity_without_hydration(backend, direction):
    memory = fixture_graph()
    request = NeighborhoodRequestV1(focal_node_ids=["aaa", "bbb"], direction=direction)
    if backend == "falkor":
        client = NativeFixture(memory)
        store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri="redis://unused"), client=client, hydrate=False)
    else:
        dataset = rdflib.Dataset()
        store = GraphDBSubstrateStore(GraphDBSubstrateStoreConfig(endpoint="http://unused"))
        store._update = dataset.update
        for node in memory._nodes.values():
            store.upsert_node(identity_key=node.node_id, node=node)
        for item in memory._edges.values():
            store.upsert_edge(identity_key=item.edge_id, edge=item)
        def select(query):
            rows = list(dataset.query(query))
            # Real SPARQL execution; cap independently from requested limit.
            return [{str(k): {"value": str(v)} for k, v in row.asdict().items()} for row in rows[:1]]
        store._select = select
    store.snapshot = lambda: pytest.fail("full hydration forbidden")
    store._cache._nodes.clear()
    store._cache._edges.clear()
    actual = store.read_neighborhood(request)
    assert not actual.degraded, actual.reason
    assert ids(actual) == ids(memory.read_neighborhood(request))
    assert actual.complete_for_request
    assert_endpoints(actual)
    assert not store._cache._nodes and not store._cache._edges


@pytest.mark.parametrize("backend", ["falkor", "sparql"])
def test_backend_failure_does_not_fallback_to_cache(backend):
    def fail(*args, **kwargs):
        raise RuntimeError("unavailable private details")
    if backend == "falkor":
        client = NativeFixture(fixture_graph())
        client.graph_query = fail
        store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri="redis://unused"), client=client, hydrate=False)
    else:
        store = GraphDBSubstrateStore(GraphDBSubstrateStoreConfig(endpoint="http://unused"))
        store._select = fail
    store._cache = fixture_graph()
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=["aaa"]))
    assert result.degraded and not result.complete_for_request
    assert not result.focal_nodes and "private" not in result.reason


def test_planner_exposes_distinct_internal_and_boundary_refs():
    store = fixture_graph()
    execution = SubstrateSemanticReadCoordinator(store=store).execute(SubstrateQueryPlanV1(
        plan_kind="diagnostic", steps=(SubstrateQueryPlanStepV1("neighborhood", {"focal_node_ids": ["aaa", "bbb"]}),)))
    result = execution.results[0]
    assert result.details["focal_edge_refs"] == ["internal-1", "internal-2"]
    assert set(result.details["boundary_edge_refs"]) == {"incoming", "outgoing"}
    assert set(result.details["neighbor_node_refs"]) == {"ccc", "ddd"}


def test_neighbors_are_fetched_in_one_batched_node_call(monkeypatch):
    """PR E: the per-neighbor round trip is gone; receipts are unchanged."""
    import orion.substrate.neighborhood as module

    store, focal = hub_fixture(40)
    request = NeighborhoodRequestV1(focal_node_ids=focal)
    expected = module.read_memory_neighborhood(store, request)
    calls = []
    original = module.read_neighborhood

    def counting(request, *, nodes, **kwargs):
        return original(request, nodes=lambda ids: calls.append(list(ids)) or nodes(ids), **kwargs)

    monkeypatch.setattr(module, "read_neighborhood", counting)
    actual = module.read_memory_neighborhood(store, request)
    assert len(calls) == 2, calls
    assert calls[0] == sorted(focal)
    assert calls[1] == [n.node_id for n in actual.neighbor_nodes] and len(calls[1]) > 1
    assert ids(actual) == ids(expected)
    assert (actual.truncated, actual.complete_for_request, actual.degraded, actual.reason) == (
        expected.truncated, expected.complete_for_request, expected.degraded, expected.reason)


def test_batched_neighbor_fetch_still_fails_closed_on_a_vanished_endpoint():
    store, focal = hub_fixture(5)
    from orion.substrate.neighborhood import read_neighborhood
    result = read_neighborhood(NeighborhoodRequestV1(focal_node_ids=focal), source_kind="test",
        nodes=lambda ids: [store._nodes[key] for key in ids if key in store._nodes and key != "neighbor-0003"],
        groups=lambda ids: sorted({(e.target.node_id, "incoming", e.predicate) for e in store._edges.values()
                                   if e.target.node_id in ids and e.source.node_id not in ids}
                                  | {(e.source.node_id, "outgoing", e.predicate) for e in store._edges.values()
                                     if e.source.node_id in ids and e.target.node_id not in ids}),
        edges=lambda ids, group, after, limit: sorted(
            [e for e in store._edges.values() if e.edge_id > after and (
                (group is None and e.source.node_id in ids and e.target.node_id in ids) or
                (group is not None and e.predicate == group[2] and (
                    (group[1] == "incoming" and e.target.node_id == group[0] and e.source.node_id not in ids) or
                    (group[1] == "outgoing" and e.source.node_id == group[0] and e.target.node_id not in ids))))],
            key=lambda e: e.edge_id)[:limit])
    assert result.degraded and not result.complete_for_request
    assert not result.neighbor_nodes and not result.boundary_edges
    assert result.reason == "unavailable:ValueError"
