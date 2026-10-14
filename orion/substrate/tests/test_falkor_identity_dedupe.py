"""Duplicate node identity: hydrate degrades per node, writers resolve durably.

Live 2026-10-14: orion_substrate_aitown held 1,237 identities on 14,022 nodes
and orion_substrate_self 1,109 on 4,640, all topic-foundry concepts/entities.
Two defects fed each other:
  1. identity resolution was cache-only, so any incomplete cache (the old
     10k-row server cap, then a failed hydrate) made the materializer mint a
     new run-scoped node for an identity already in the graph;
  2. one duplicate identity failed the whole hydrate, so the hub booted with
     an empty cache and the next ingest duplicated every identity again.
"""
from orion.core.schemas.cognitive_substrate import (
    SubstrateGraphRecordV1,
)
from orion.substrate.evals.complete_hydration_fixture import CappedClient, build, topic_concept
from orion.substrate.falkor_store import NODE_BY_IDENTITY_CYPHER, SUBSTRATE_INDEXES
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.reconcile import SubstrateIdentityResolver


_topic_concept = topic_concept


def test_duplicate_identity_hydrates_every_node_and_resolves_to_smallest_node_id(caplog):
    client = CappedClient()
    client._hydrate_node_rows[3].update(identity_key="node:5")  # node3 and node5 share node:5
    with caplog.at_level("WARNING"):
        store = build(client)
    state = store.snapshot()
    assert state.scan_receipt.complete and not state.scan_receipt.stale
    assert len(state.nodes) == 7  # nothing dropped
    assert state.scan_receipt.duplicate_node_identities == 1
    assert store.hydrate_duplicate_node_identities == 1
    assert state.node_identity_index["node:5"] == "node3"
    # the non-canonical duplicate keeps its own identity for later writes
    assert store.get_identity_key_by_node_id("node5") == "node:5"
    assert "falkor_substrate_hydrate_duplicate_identities" in caplog.text
    assert "falkor_substrate_hydrate_failed" not in caplog.text


def test_cache_miss_falls_back_to_durable_lookup_and_caches_the_node():
    client = CappedClient()
    store = build(client)
    # Simulate an incomplete cache (failed boot hydrate, or another writer).
    store._cache = type(store._cache)()
    assert store.get_node_by_id("node4") is None
    assert store.get_node_id_by_identity("node:4") == "node4"
    assert any(q == NODE_BY_IDENTITY_CYPHER for q, _ in client.calls)
    assert store.get_node_by_id("node4") is not None
    assert store.identity_durable_hits_total == 1


def test_unknown_identity_returns_none():
    store = build(CappedClient())
    store._cache = type(store._cache)()
    assert store.get_node_id_by_identity("concept|world||label:never seen") is None
    assert store.identity_durable_lookups_total == 1


def test_durable_lookup_failure_degrades_to_none_and_logs(caplog):
    client = CappedClient()
    store = build(client)
    store._cache = type(store._cache)()

    def boom(cypher, params=None):
        raise ConnectionError("down")

    client.graph_query = boom
    with caplog.at_level("WARNING"):
        assert store.get_node_id_by_identity("node:1") is None
    assert "falkor_substrate_identity_lookup_failed" in caplog.text


def test_materializer_merges_a_new_run_into_the_durable_identity_instead_of_minting():
    """The live defect end to end: a second topic-foundry run carries a new
    run-scoped node_id for a label already in the graph, and the store's
    cache does not hold it. It must merge, not create."""
    first = _topic_concept("sub-concept-topicfoundry-run1-53", "Orion's coffee and questions")
    key = SubstrateIdentityResolver().canonical_node_key(first)
    client = CappedClient(nodes=0, edges=0)
    from orion.substrate.falkor_codec import encode_node_properties
    client._hydrate_node_rows.append(encode_node_properties(first, identity_key=key))
    store = build(client)
    store._cache = type(store._cache)()  # boot hydrate "failed": empty cache

    second = _topic_concept("sub-concept-topicfoundry-run2-53", "Orion's coffee and questions")
    record = SubstrateGraphRecordV1(graph_id="sub-graph-topicfoundry-run2", anchor_scope="world", nodes=[second], edges=[])
    materializer = SubstrateGraphMaterializer(store=store, identity_resolver=SubstrateIdentityResolver(store=store))
    result = materializer.apply_record(record)
    assert result.nodes_created == 0 and result.nodes_merged == 1
    writes = [p for q, p in client.calls if q.startswith("MERGE (n:SubstrateNode")]
    assert [p["node_id"] for p in writes] == ["sub-concept-topicfoundry-run1-53"]


def test_identity_key_is_indexed():
    assert ("SubstrateNode", "identity_key") in SUBSTRATE_INDEXES


def test_a_write_to_the_larger_duplicate_does_not_move_the_identity():
    """Review 2026-10-14: InMemory upsert points an identity at the last
    writer; a tick re-saving the larger copy would flip canonical."""
    from orion.substrate.falkor_codec import decode_node

    client = CappedClient()
    client._hydrate_node_rows[3].update(identity_key="node:5")
    store = build(client)
    assert store.get_node_id_by_identity("node:5") == "node3"
    dup = decode_node(client._hydrate_node_rows[5])
    store.upsert_node(identity_key="node:5", node=dup)
    assert store.get_node_id_by_identity("node:5") == "node3"
    assert store.get_identity_key_by_node_id("node5") == "node:5"


def test_durable_hit_during_an_inflight_scan_survives_the_swap():
    client = CappedClient()
    store = build(client)
    store._cache = type(store._cache)()
    store._scan_journal = []  # a hydrate is mid-scan
    assert store.get_node_id_by_identity("node:2") == "node2"
    assert [entry[2].node_id for entry in store._scan_journal] == ["node2"]
