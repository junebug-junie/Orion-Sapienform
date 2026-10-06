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
    assert all("ORDER BY object_id LIMIT $page_size" in q for q, _ in client.calls)
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
    (lambda c: c._hydrate_node_rows.append(dict(c._hydrate_node_rows[0])), "duplicate node_id"),
    (lambda c: c._hydrate_edge_rows.append(dict(c._hydrate_edge_rows[0])), "duplicate edge_id"),
    (lambda c: c._hydrate_node_rows[1].update(identity_key="node:0"), "duplicate node identity"),
    (lambda c: c._hydrate_edge_rows[0].update(source_id="missing"), "endpoint"),
    (lambda c: c._hydrate_edge_rows[0].update(source_kind="entity"), "endpoint"),
    (lambda c: c._hydrate_node_rows[0].update(node_kind="unknown"), "unsupported"),
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


def test_local_write_during_refresh_keeps_writer_cache_and_retries():
    client = CappedClient()
    store = build(client)
    original = client.graph_query
    mutated = False
    def query(cypher, params=None):
        nonlocal mutated
        result = original(cypher, params)
        if not mutated:
            mutated = True
            store.upsert_node(identity_key="new", node=_concept(node_id="new"))
            client._hydrate_node_rows.append(_hydrated_node_row("new", "new"))
        return result
    client.graph_query = query
    store._hydrate_from_durable()
    assert not store.last_hydrate_ok
    assert store.get_node_by_id("new") is not None
    assert store.snapshot().scan_receipt.complete
    assert store.get_node_by_id("new") is not None


def test_detected_external_insertion_with_missing_endpoint_rejects_scan():
    client = CappedClient()
    store = build(client)
    original = client.graph_query
    def query(cypher, params=None):
        if "RETURN e.edge_id" in cypher:
            client._hydrate_edge_rows = [edge_row(0, target="inserted-after-node-scan")]
        return original(cypher, params)
    client.graph_query = query
    previous = store._cache
    store._hydrate_from_durable()
    assert store._cache is previous and not store.last_hydrate_ok


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
