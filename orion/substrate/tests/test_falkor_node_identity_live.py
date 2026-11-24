"""upsert_node never mints a second node for one node_id (#3512), on a real FalkorDB.

Only a real engine can prove this: the bug was MERGE's label match, which the
recording test doubles never execute. Throwaway graph from ORION_TEST_FALKOR_URI;
CI runs it in .github/workflows/substrate-neighborhood.yml, which fails on a skip.
"""

from __future__ import annotations

import os
import uuid

import pytest

from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
from orion.substrate.tests.test_assertion_core import concept, edge, entity

_FALKOR_URI = os.getenv("ORION_TEST_FALKOR_URI", "").strip()
pytestmark = pytest.mark.skipif(not _FALKOR_URI, reason="ORION_TEST_FALKOR_URI not set (throwaway FalkorDB)")


@pytest.fixture()
def graph():
    from orion.graph.falkor_client import RedisGraphQueryClient

    name = f"t_identity_{uuid.uuid4().hex[:10]}"
    client = RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=name)
    yield name, client
    try:
        client.graph_query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass


def _store(name: str, client, *, hydrate: bool = False) -> FalkorSubstrateStore:
    return FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=_FALKOR_URI, graph_name=name),
                                client=client, hydrate=hydrate)


def _rows(client, node_id: str) -> list:
    return client.graph_query(
        "MATCH (n:SubstrateNode {node_id: $node_id}) RETURN n.node_id AS node_id, labels(n) AS labels",
        {"node_id": node_id})


def test_a_node_upserted_after_its_edge_placeholder_is_one_node(graph):
    """upsert_edge MERGEs bare SubstrateNode placeholders for its endpoints. The
    old typed MERGE (SubstrateNode:Concept) never matched one, so the node's own
    upsert minted a twin and the next hydrate failed on 'duplicate node_id'."""
    name, client = graph
    store = _store(name, client)
    store.upsert_node(identity_key="k|ent-a", node=entity("ent-a", "a"))
    store.upsert_edge(identity_key="e1", edge=edge("edge-1", ("ent-a", "entity"), ("con-b", "concept")))
    store.upsert_node(identity_key="k|con-b", node=concept("con-b", "b"))
    assert len(_rows(client, "con-b")) == 1
    state = _store(name, client, hydrate=True).snapshot()
    assert state.scan_receipt.complete, state.scan_receipt.reason
    assert state.scan_receipt.integrity_anomalies == {}
    assert state.nodes["con-b"].node_kind == "concept" and "edge-1" in state.edges


def test_a_node_missing_its_type_label_is_updated_in_place(graph):
    """Live 2026-11-24 (aitown 49919): a node read Concept from labels() but was
    absent from the Concept label scan, so the typed MERGE missed it and wrote a
    twin. A node truly lacking its type label is the closest real-engine stand-in."""
    name, client = graph
    store = _store(name, client)
    client.graph_query("CREATE (:SubstrateNode {node_id: 'con-c', node_kind: 'concept'})")
    store.upsert_node(identity_key="k|con-c", node=concept("con-c", "relabelled"))
    rows = _rows(client, "con-c")
    assert len(rows) == 1
    assert "Concept" in str(rows[0])


def test_legacy_payload_twin_of_a_canonical_node_is_removed_and_a_lone_one_is_converted(graph):
    name, client = graph
    payload = concept("twin", "legacy twin").model_dump_json()
    lone = concept("lone", "legacy lone").model_dump_json()
    client.graph_query(
        "CREATE (:SubstrateNode {node_id: 'twin', payload_json: $p}), "
        "(:SubstrateNode {node_id: 'lone', payload_json: $q})", {"p": payload, "q": lone})
    _store(name, client).upsert_node(identity_key="k|twin", node=concept("twin", "canonical"))
    # The canonical upsert matched the legacy row too (one node_id), so it is
    # converted in place, not twinned.
    assert len(_rows(client, "twin")) == 1
    # A legacy row next to an already-native node with the same node_id.
    _store(name, client).upsert_node(identity_key="k|pair", node=concept("pair", "canonical pair"))
    client.graph_query(
        "CREATE (:SubstrateNode {node_id: 'pair', payload_json: $p})",
        {"p": concept("pair", "legacy pair").model_dump_json()})
    assert len(_rows(client, "pair")) == 2
    store = _store(name, client, hydrate=True)  # hydrate runs the legacy migration
    assert store.last_hydrate_ok, store.last_scan_receipt.reason
    for node_id in ("twin", "lone", "pair"):
        rows = client.graph_query(
            "MATCH (n:SubstrateNode {node_id: $node_id}) RETURN n.payload_json IS NULL AS native",
            {"node_id": node_id})
        assert len(rows) == 1, (node_id, rows)
    # The stale legacy payload never overwrote the native node.
    assert store.snapshot().nodes["pair"].label == "canonical pair"
    labels = client.graph_query("MATCH (n:SubstrateNode {node_id: 'pair'}) RETURN n.label AS label")
    assert "canonical pair" in str(labels) and "legacy pair" not in str(labels)
