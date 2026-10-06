"""Shared assertion core against a real throwaway FalkorDB (ORION_TEST_FALKOR_URI).

The assertion-state gate runs as Cypher (OPTIONAL MATCH on e.assertion_id), so only a
real Falkor can prove it; the in-memory lane is test_assertion_core.py. CI runs this
in .github/workflows/substrate-neighborhood.yml, which fails on any skip.
"""

from __future__ import annotations

import os
import uuid

import pytest

from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
from orion.substrate.neighborhood import NeighborhoodRequestV1
from orion.substrate.tests.test_assertion_core import FENCED, assertion, edge, entity

_FALKOR_URI = os.getenv("ORION_TEST_FALKOR_URI", "").strip()
pytestmark = pytest.mark.skipif(not _FALKOR_URI, reason="ORION_TEST_FALKOR_URI not set (throwaway FalkorDB)")


@pytest.fixture()
def graph():
    from orion.graph.falkor_client import RedisGraphQueryClient

    name = f"t_assertion_{uuid.uuid4().hex[:10]}"
    client = RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=name)
    yield name, client
    try:
        client.graph_query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass


def _write(client, *, state: str, revision: int, edge_revision: int) -> FalkorSubstrateStore:
    store = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=_FALKOR_URI, graph_name="unused"),
                                 client=client, hydrate=False)
    for node in (entity("quill", "quill", producer=FENCED, scope="juniper"),
                 entity("retreat", "spring retreat", producer=FENCED, scope="juniper"),
                 entity("legacy-n", "legacy", scope="juniper"),
                 assertion("as1", state=state, revision=revision)):
        store.upsert_node(identity_key=f"k|{node.node_id}", node=node)
    store.upsert_edge(identity_key="projection|as1", edge=edge(
        "proj", ("quill", "entity"), ("retreat", "entity"), edge_role="semantic_projection",
        assertion_id="as1", assertion_revision=edge_revision))
    store.upsert_edge(identity_key="leg", edge=edge("leg", ("quill", "entity"), ("legacy-n", "entity"),
                                                    predicate="associated_with"))
    store.upsert_edge(identity_key="struct", edge=edge("struct", ("as1", "assertion"), ("quill", "entity"),
                                                       predicate="assertion_subject",
                                                       edge_role="assertion_structure"))
    return store


@pytest.mark.parametrize("state,revision,edge_revision,walks", [
    ("provisional", 1, 1, True),
    ("canonical", 2, 2, True),
    ("rejected", 2, 2, False),
    ("provisional", 2, 1, False),
])
def test_falkor_walks_a_projection_only_while_its_assertion_is_accepted(graph, state, revision, edge_revision, walks):
    _, client = graph
    store = _write(client, state=state, revision=revision, edge_revision=edge_revision)
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("quill",)))
    assert not result.degraded, result.reason
    edges = {e.edge_id for e in result.boundary_edges}
    assert ("proj" in edges) is walks
    assert "leg" in edges and "struct" not in edges
    if walks:
        proj = next(e for e in result.boundary_edges if e.edge_id == "proj")
        assert (proj.edge_role, proj.assertion_id, proj.assertion_revision) == ("semantic_projection", "as1",
                                                                                edge_revision)


def test_assertion_nodes_and_edge_roles_survive_a_full_hydration(graph):
    name, client = graph
    _write(client, state="provisional", revision=1, edge_revision=1)
    fresh = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=_FALKOR_URI, graph_name=name), client=client)
    state = fresh.snapshot()
    # metadata differs only by the store's activation_decayed_at stamp.
    expected = assertion("as1", state="provisional", revision=1).model_dump(exclude={"metadata"})
    assert state.nodes["as1"].model_dump(exclude={"metadata"}) == expected
    assert state.edges["proj"].edge_role == "semantic_projection"
    assert state.edges["struct"].edge_role == "assertion_structure"
    assert state.edges["leg"].edge_role == "legacy_unreviewed"
