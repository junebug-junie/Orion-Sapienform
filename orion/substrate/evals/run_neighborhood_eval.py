"""Hub-heavy budget/coverage replay; diagnostics only, never cognition inputs."""
import json
from time import perf_counter

from orion.substrate.evals.neighborhood_fixture import hub_fixture
from orion.substrate.neighborhood import NeighborhoodRequestV1


def main():
    store, focal = hub_fixture(3355)
    receipts = []
    for internal, boundary, neighbors in [(12, 16, 16), (4, 3, 3), (0, 0, 0), (12, 32, 8)]:
        started = perf_counter()
        result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=focal,
            internal_edge_limit=internal, boundary_edge_limit=boundary, neighbor_node_limit=neighbors))
        node_ids = {n.node_id for n in result.focal_nodes + result.neighbor_nodes}
        assert len(result.internal_edges) == min(4, internal)
        assert len(result.boundary_edges) <= boundary and len(result.neighbor_nodes) <= neighbors
        assert result.truncated and not result.degraded
        assert all(e.source.node_id in node_ids and e.target.node_id in node_ids for e in result.internal_edges + result.boundary_edges)
        if boundary >= 2 and neighbors >= 2:
            assert "small-in" in {e.edge_id for e in result.boundary_edges}
        if boundary >= 4 and neighbors >= 3:
            assert {"small-in", "small-out"} <= {e.edge_id for e in result.boundary_edges}
        receipts.append({"budgets": [internal, boundary, neighbors],
            "returned": [len(result.internal_edges), len(result.boundary_edges), len(result.neighbor_nodes)],
            "truncated": result.truncated, "endpoint_integrity": True,
            "elapsed_ms": round((perf_counter()-started)*1000, 2)})
    print(json.dumps({"fixture_boundary_edges": 3357, "checks": receipts}, indent=2))


if __name__ == "__main__":
    main()
