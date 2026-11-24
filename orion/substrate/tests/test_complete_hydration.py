"""Complete scan contract: server caps, integrity, retry and mutation failures."""
import pytest

from orion.substrate.evals.complete_hydration_fixture import (
    CappedClient, build, edge_row, _concept, _hydrated_node_row,
)


@pytest.mark.parametrize("page_size,cap", [(1,9), (3,2), (1000,2), (1000,1000)])
def test_short_pages_are_not_terminal(page_size, cap):
    client = CappedClient(cap=cap)
    store = build(client, hydration_page_size=page_size)
    state = store.snapshot()
    assert len(state.nodes) == 7 and len(state.edges) == 11
    assert state.scan_receipt.complete and not state.scan_receipt.stale
    assert state.scan_receipt.consistency == "non_atomic_keyset"
    assert all("WITH * ORDER BY id(" in q and "LIMIT $page_size RETURN" in q for q, _ in client.calls
               if "AS node_count" not in q)
    assert {p["after_id"] for q, p in client.calls if "RETURN e.edge_id" in q} >= {-1,10}


def test_more_than_deployed_server_cap():
    store = build(CappedClient(nodes=2, edges=10007, cap=613))
    assert store.last_hydrate_ok
    assert len(store.snapshot().edges) == 10007


def test_mid_scan_failure_preserves_cache_and_success_cursor_and_retries():
    client = CappedClient()
    store = build(client)
    old_cache, timestamp, generation = store._cache, store._last_snapshot_at, store._last_snapshot_generation
    old_success = store.last_scan_receipt.last_successful_refresh_at
    client.fail = True
    store._write_generation += 1
    state = store.snapshot()
    assert store._cache is old_cache
    assert (store._last_snapshot_at, store._last_snapshot_generation) == (timestamp, generation)
    assert not state.scan_receipt.complete and state.scan_receipt.stale
    assert state.scan_receipt.last_successful_refresh_at == old_success
    calls = len(client.calls)
    store.snapshot()
    assert len(client.calls) > calls  # no false success ceiling suppresses retry
    client.fail = False
    assert store.snapshot().scan_receipt.complete
    assert store._cache is not old_cache


def test_failed_boot_does_not_mark_empty_cache_fresh():
    client = CappedClient()
    client.fail = True
    store = build(client, snapshot_force_refresh_ceiling_sec=0)
    assert store._last_snapshot_at is None and store._last_snapshot_generation == -1
    client.fail = False
    assert len(store.snapshot().nodes) == 7


@pytest.mark.parametrize("mutation,reason", [
    (lambda c: c._hydrate_node_rows[0].update(node_id=None), "invalid"),
    (lambda c: c._hydrate_node_rows[0].update(anchor_scope="invalid"), "validation"),
    (lambda c: c._hydrate_edge_rows[0].update(edge_id=None), "invalid"),
    (lambda c: c._hydrate_edge_rows[1].update(identity_key="edge:0", predicate="supports"), "incompatible edge identity"),
])
def test_bad_rows_never_replace_good_cache(mutation, reason):
    client = CappedClient()
    store = build(client)
    cache = store._cache
    mutation(client)
    store._hydrate_from_durable()
    assert store._cache is cache
    assert not store.last_scan_receipt.complete
    assert reason in store.last_scan_receipt.reason


def test_a_shape_newer_than_this_reader_is_skipped_and_counted_not_fatal():
    """#2515 review: forward tolerance. An unknown node kind is skipped with every edge
    touching it; an unknown predicate or edge role is skipped; the rest hydrates."""
    client = CappedClient()
    gone = client._hydrate_node_rows[0]["node_id"]
    client._hydrate_node_rows[0].update(node_kind="future_kind")
    client._hydrate_edge_rows[-1].update(predicate="future_predicate")
    expected = sum(1 for r in client._hydrate_edge_rows
                   if gone in (r.get("source_id"), r.get("target_id")) or r["predicate"] == "future_predicate")
    store = build(client)
    state = store.snapshot()
    assert state.scan_receipt.complete
    assert store.hydrate_skipped_unknown_nodes == 1
    assert store.hydrate_skipped_unknown_edges == expected >= 1
    assert gone not in state.nodes
    assert len(state.nodes) == 6


def test_parallel_edges_with_compatible_lookup_alias_are_all_preserved():
    client = CappedClient(edges=12)
    for row in client._hydrate_edge_rows:
        row["identity_key"] = "parallel"
    store = build(client)
    state = store.snapshot()
    assert len(state.edges) == 12
    assert state.edge_identity_index["parallel"] == "edge0"
    assert state.scan_receipt.edge_identity_aliases == 11


@pytest.mark.parametrize("bad_result", [None, {}, [None], [[1]], [{"_positional": [1]}],
    [{"object_id": -1}], [{"object_id": "1"}], [{"object_id": True}]])
def test_malformed_pages_are_failures_not_empty_success(bad_result):
    class BadClient:
        def graph_query(self, *_args, **_kw):
            return bad_result
    store = build(BadClient())
    assert not store.last_hydrate_ok
    assert store._last_snapshot_at is None


def test_nonadvancing_cursor_fails_instead_of_looping():
    class StuckClient(CappedClient):
        def graph_query(self, cypher, params=None):
            if "RETURN n.node_id" in cypher:
                return [dict(self._hydrate_node_rows[0], object_id=0)]
            return super().graph_query(cypher, params)
    store = build(StuckClient())
    assert not store.last_hydrate_ok
    assert "cursor" in store.last_scan_receipt.reason


def test_a_decreasing_object_id_inside_a_page_still_fails():
    class BackwardsClient(CappedClient):
        def graph_query(self, cypher, params=None):
            if "RETURN n.node_id" in cypher and params and params.get("after_id") == -1:
                return [dict(self._hydrate_node_rows[0], object_id=5),
                        dict(self._hydrate_node_rows[1], object_id=3)]
            return super().graph_query(cypher, params)
    store = build(BackwardsClient())
    assert not store.last_hydrate_ok
    assert "cursor" in store.last_scan_receipt.reason


def test_local_write_during_refresh_is_replayed_not_aborted():
    """A local write mid-scan used to abort the refresh ("local mutation during
    scan; retry required"). It is now journaled and replayed onto the staged
    cache, so the refresh completes and holds both the scan and the write."""
    client = CappedClient()
    store = build(client)
    original = client.graph_query
    mutated = False
    def query(cypher, params=None):
        nonlocal mutated
        result = original(cypher, params)
        if not mutated:
            mutated = True
            # Not added to the durable rows: the scan must not need to see it.
            store.upsert_node(identity_key="new", node=_concept(node_id="new"))
        return result
    client.graph_query = query
    store._write_generation += 1
    state = store.snapshot()
    assert store.last_hydrate_ok
    assert state.scan_receipt.complete and not state.scan_receipt.stale
    assert "new" in state.nodes and state.node_identity_index["new"] == "new"
    assert {f"node{i}" for i in range(7)} <= set(state.nodes)
    assert store._last_snapshot_generation == store._write_generation
    assert store._scan_journal is None
    assert store.hydrate_replayed_writes_total == 1


def test_continuous_local_writes_no_longer_abort_every_refresh():
    """Regression for the live abort loop (orion-athena-substrate-runtime,
    2026-10-06: ~1 refresh in 2 failed). A process whose tick loops write
    during every scan never completed a refresh under the abort rule; every
    one now completes."""
    client = CappedClient()
    store = build(client)
    original = client.graph_query
    counter = {"n": 0}
    def query(cypher, params=None):
        result = original(cypher, params)
        # Bounded per scan: appended durable rows extend the scan, so an
        # unbounded writer would keep it running forever.
        if params and params.get("after_id") == -1:
            counter["n"] += 1
            n = counter["n"]
            store.upsert_node(identity_key=f"w:{n}", node=_concept(node_id=f"w{n}"))
            client._hydrate_node_rows.append(_hydrated_node_row(f"w{n}", f"w:{n}"))
        return result
    client.graph_query = query
    ok_before, failed_before = store.hydrate_ok_total, store.hydrate_failed_total
    for i in range(5):
        # A write between snapshots forces a real refresh, as in the runtime.
        store.upsert_node(identity_key=f"between:{i}", node=_concept(node_id=f"between{i}"))
        client._hydrate_node_rows.append(_hydrated_node_row(f"between{i}", f"between:{i}"))
        state = store.snapshot()
        assert state.scan_receipt.complete, state.scan_receipt.reason
        assert {f"w{i}" for i in range(1, counter["n"] + 1)} <= set(state.nodes)
    assert store.hydrate_ok_total - ok_before == 5
    assert store.hydrate_failed_total == failed_before


def test_replayed_edge_and_skip_keys_merge_against_fresh_scan():
    """Journaled edges land in the swapped cache, and a journaled write that
    skips an externally owned key keeps the freshly scanned durable value, not
    the old cache's or the caller's copy."""
    from orion.substrate.falkor_codec import EXTERNALLY_OWNED_METADATA_KEYS, encode_node_properties
    from orion.core.schemas.cognitive_substrate import NodeRefV1, SubstrateEdgeV1
    client = CappedClient()
    durable = _concept(node_id="node0").model_copy(update={"metadata": {"prediction_error": 0.9}})
    client._hydrate_node_rows[0] = encode_node_properties(durable, identity_key="node:0")
    store = build(client)
    original = client.graph_query
    done = False
    def query(cypher, params=None):
        nonlocal done
        result = original(cypher, params)
        if not done:
            done = True
            stale = _concept(node_id="node0").model_copy(update={"metadata": {"prediction_error": 0.1}})
            store.upsert_node(identity_key="node:0", node=stale, skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS)
            base = _concept()
            store.upsert_edge(identity_key="local-edge", edge=SubstrateEdgeV1(
                edge_id="local-edge", source=NodeRefV1(node_id="node0", node_kind="concept"),
                target=NodeRefV1(node_id="node2", node_kind="concept"), predicate="associated_with",
                temporal=base.temporal, provenance=base.provenance))
        return result
    client.graph_query = query
    store._write_generation += 1
    state = store.snapshot()
    assert state.scan_receipt.complete
    assert state.nodes["node0"].metadata.get("prediction_error") == 0.9
    assert "local-edge" in state.edges


def test_failed_refresh_stops_journaling():
    client = CappedClient()
    store = build(client)
    client.fail = True
    store._write_generation += 1
    store.snapshot()
    assert not store.last_hydrate_ok and store._scan_journal is None
    store.upsert_node(identity_key="after", node=_concept(node_id="after"))
    assert store._scan_journal is None
    assert store.hydrate_failed_total >= 1


def test_detected_external_insertion_with_missing_endpoint_skips_the_edge_not_the_scan():
    """An edge written after the node pass read its endpoint is skipped and
    counted; the next refresh picks it up. Before #3512 it failed the scan."""
    client = CappedClient()
    store = build(client)
    original = client.graph_query
    def query(cypher, params=None):
        if "RETURN e.edge_id" in cypher:
            client._hydrate_edge_rows = [edge_row(0, target="inserted-after-node-scan"), edge_row(1)]
        return original(cypher, params)
    client.graph_query = query
    store._hydrate_from_durable()
    assert store.last_hydrate_ok
    assert set(store._cache.snapshot().edges) == {"edge1"}
    assert store.last_scan_receipt.integrity_anomalies == {"bad_endpoint_edge": 1}


def test_read_only_legacy_scan_decodes_without_rewriting():
    client = CappedClient(nodes=0, edges=0, cap=2)
    client.read_only = True
    client._hydrate_legacy_node_rows = [dict(payload_json=_concept(node_id=f"l{i}").model_dump_json(),
                                            identity_key=f"legacy:{i}") for i in range(5)]
    store = build(client)
    assert store.last_hydrate_ok and store.last_hydrate_node_count == 5
    assert all(q.startswith("MATCH") and "MERGE" not in q and "DELETE" not in q for q, _ in client.calls)


def test_bad_legacy_payload_is_not_silently_skipped():
    client = CappedClient()
    client._hydrate_legacy_node_rows = [dict(payload_json="{}", identity_key="broken")]
    assert not build(client).last_hydrate_ok


@pytest.mark.parametrize("size", [0, -1, 10001])
def test_page_size_is_bounded(size):
    with pytest.raises(ValueError):
        build(CappedClient(), hydration_page_size=size)


@pytest.mark.parametrize("fail_rewrite", [False, True])
def test_writable_legacy_rewrite_preserves_staged_lookup_representative(fail_rewrite):
    from orion.substrate.falkor_codec import decode_edge
    client = CappedClient(nodes=2, edges=1)
    client._hydrate_edge_rows[0]["identity_key"] = "shared"
    client._hydrate_legacy_edge_rows = [dict(payload_json=decode_edge(edge_row(9)).model_dump_json(),
                                           identity_key="shared")]
    original = client.graph_query
    def query(cypher, params=None):
        if fail_rewrite and "MERGE" in cypher:
            raise ConnectionError("rewrite outage")
        return original(cypher, params)
    client.graph_query = query
    store = build(client)
    assert store.last_hydrate_ok
    assert store.get_edge_id_by_identity("shared") == "edge0"
    assert len(store._cache.snapshot().edges) == 2
    assert store.last_scan_receipt.edge_identity_aliases == 1


@pytest.mark.parametrize("malformed", [None, [None], [1], {"unexpected": "shape"}])
def test_real_client_parser_cannot_hide_malformed_refresh(malformed):
    from types import SimpleNamespace
    from orion.graph.falkor_client import RedisGraphQueryClient
    client = RedisGraphQueryClient.__new__(RedisGraphQueryClient)
    rows = CappedClient()
    class Graph:
        fail = False
        def query(self, cypher, params=None):
            return SimpleNamespace(header=[], result_set=malformed if self.fail else rows.graph_query(cypher, params))
    client._graph = Graph()
    store = build(client)
    assert store.last_hydrate_ok
    old_cache, old_cursor = store._cache, store._last_snapshot_at
    client._graph.fail = True
    store._write_generation += 1
    state = store.snapshot()
    assert not state.scan_receipt.complete and state.scan_receipt.stale
    assert store._cache is old_cache and store._last_snapshot_at == old_cursor
    assert "malformed Falkor result" in state.scan_receipt.reason


def test_replayed_edge_with_missing_endpoint_is_not_cached():
    """A journaled edge whose endpoint the fresh scan does not contain (deleted
    elsewhere mid-scan) is dropped rather than cached dangling."""
    from orion.core.schemas.cognitive_substrate import NodeRefV1, SubstrateEdgeV1
    client = CappedClient()
    store = build(client)
    original = client.graph_query
    done = False
    def query(cypher, params=None):
        nonlocal done
        result = original(cypher, params)
        if not done:
            done = True
            base = _concept()
            store.upsert_edge(identity_key="dangling", edge=SubstrateEdgeV1(
                edge_id="dangling", source=NodeRefV1(node_id="node0", node_kind="concept"),
                target=NodeRefV1(node_id="gone", node_kind="concept"), predicate="associated_with",
                temporal=base.temporal, provenance=base.provenance))
        return result
    client.graph_query = query
    store._write_generation += 1
    state = store.snapshot()
    assert state.scan_receipt.complete
    assert "dangling" not in state.edges


# ── #3512: corrupt durable rows are skipped and counted, never fatal ─────────

@pytest.mark.parametrize("mutation,kind", [
    (lambda c: c._hydrate_node_rows.append(dict(c._hydrate_node_rows[0])), "duplicate_node_id"),
    (lambda c: c._hydrate_edge_rows.append(dict(c._hydrate_edge_rows[0])), "duplicate_edge_id"),
    (lambda c: c._hydrate_edge_rows[0].update(source_id="missing"), "bad_endpoint_edge"),
    (lambda c: c._hydrate_edge_rows[0].update(source_kind="entity"), "bad_endpoint_edge"),
])
def test_a_corrupt_row_is_skipped_and_counted_and_the_rest_hydrates(mutation, kind):
    client = CappedClient()
    mutation(client)
    store = build(client)
    state = store.snapshot()
    assert store.last_hydrate_ok and state.scan_receipt.complete
    assert len(state.nodes) == 7
    assert len(state.edges) == (10 if kind == "bad_endpoint_edge" else 11)
    assert state.scan_receipt.integrity_anomalies == {kind: 1}
    assert store.hydrate_integrity_anomalies == {kind: 1}


@pytest.mark.parametrize("phantom_first", [True, False])
def test_3512_one_edge_read_back_through_a_phantom_adjacency_entry(phantom_first):
    """Live 2026-11-24, orion_substrate: edge object 66265 (observed_in, entity ->
    evidence) came back twice from one scan, once to its real evidence target and
    once to an unrelated Concept node. Hub logged 'missing or mismatched
    endpoint: sub-concept-wp-read-8b7af82ab1ff79e8' and held 0 of 19,142 nodes.
    The real copy hydrates; the phantom is counted, in either scan order."""
    client = CappedClient()
    rows = client._hydrate_edge_rows
    for i, row in enumerate(rows):
        row["object_id"] = i
    # Same edge object (same object_id), second traversal to the wrong node.
    phantom = dict(rows[0], target_id="node5", target_kind="evidence")
    rows.insert(0, phantom) if phantom_first else rows.insert(1, phantom)
    store = build(client)
    state = store.snapshot()
    assert store.last_hydrate_ok and len(state.nodes) == 7 and len(state.edges) == 11
    edge = state.edges["edge0"]
    assert (edge.source.node_id, edge.target.node_id) == ("node0", "node1")
    assert state.scan_receipt.integrity_anomalies == {"bad_endpoint_edge": 1}


def test_3512_two_durable_rows_with_one_node_id_keep_the_newer_row_and_both_edge_sets():
    """Live 2026-11-24, orion_substrate_aitown: two Concept rows held
    sub-concept-topicfoundry-...-33 (object ids 49919 and 50246). Hub logged
    'duplicate node_id' and held 0 of 52,160 nodes. The row writers still touch
    (newer observed_at) wins; edges to either row stay attached by node_id."""
    client = CappedClient()
    older = client._hydrate_node_rows[1]
    newer = dict(older, label="newer copy", observed_at="2026-11-24T21:33:43+00:00")
    client._hydrate_node_rows.insert(0, newer)  # scan order must not decide
    store = build(client)
    state = store.snapshot()
    assert store.last_hydrate_ok and len(state.nodes) == 7 and len(state.edges) == 11
    assert state.nodes["node1"].label == "newer copy"
    assert state.scan_receipt.integrity_anomalies == {"duplicate_node_id": 1}
    assert state.scan_receipt.duplicate_node_identities == 0


def test_many_corrupt_rows_mean_a_broken_scan_and_keep_the_last_good_cache():
    """A poisoned plan that shifts every edge's endpoint kind must not swap in
    an edgeless graph as a good refresh."""
    client = CappedClient(edges=80)
    store = build(client)
    cache = store._cache
    for row in client._hydrate_edge_rows:
        row["target_kind"] = "evidence"
    store._write_generation += 1
    state = store.snapshot()
    assert store._cache is cache and not store.last_hydrate_ok
    assert "integrity anomalies over budget" in state.scan_receipt.reason
    assert state.scan_receipt.integrity_anomalies == {"bad_endpoint_edge": 80}


def test_a_small_graph_whose_every_edge_is_corrupt_still_fails_closed():
    """Review finding: the 50-row floor alone let an all-bad small graph pass."""
    client = CappedClient(edges=11)
    store = build(client)
    cache = store._cache
    for row in client._hydrate_edge_rows:
        row["target_kind"] = "evidence"
    store._write_generation += 1
    state = store.snapshot()
    assert store._cache is cache and not store.last_hydrate_ok
    assert "integrity anomalies over budget" in state.scan_receipt.reason


def test_a_legacy_payload_twin_never_overwrites_its_native_node():
    """Review finding: a legacy payload row beside a native row with the same
    node_id must not win the cache or be upserted over the native node; it is
    only deleted."""
    client = CappedClient()
    native_label = client._hydrate_node_rows[2]["label"]
    stale = _concept(node_id="node2").model_copy(update={"label": "stale legacy label"})
    client._hydrate_legacy_node_rows = [dict(payload_json=stale.model_dump_json(), identity_key="node:2")]
    store = build(client)
    state = store.snapshot()
    assert store.last_hydrate_ok
    assert state.nodes["node2"].label == native_label
    assert state.scan_receipt.integrity_anomalies == {"legacy_payload_twin": 1}
    writes = [(q, p) for q, p in client.calls if q.startswith("MERGE (n:SubstrateNode")]
    assert not any(p.get("node_id") == "node2" for _q, p in writes)
    deletes = [p for q, p in client.calls if "DETACH DELETE" in q]
    assert deletes == [{"node_id": "node2"}]


def test_a_newer_twin_keeps_the_smallest_node_id_canonical_for_a_shared_identity():
    """Review finding: replacing a duplicate row must not bypass the
    smallest-node_id rule for an identity another node also holds."""
    client = CappedClient()
    rows = client._hydrate_node_rows
    rows[1]["identity_key"] = "shared"   # node1
    rows[4]["identity_key"] = "shared"   # node4
    rows.append(dict(rows[4], observed_at="2026-11-24T21:33:43+00:00"))  # newer node4 twin
    store = build(client)
    state = store.snapshot()
    assert store.last_hydrate_ok
    assert state.node_identity_index["shared"] == "node1"
    assert state.scan_receipt.integrity_anomalies == {"duplicate_node_id": 1}
