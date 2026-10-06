"""Recall never loads the complete substrate graph (2026-10-06).

concept_region used to read from FalkorSubstrateStore's in-process copy of
the whole graph, which takes 17-25s to build since PR #2500. These tests fail
if any part of recall's concept_region path -- store construction, the
region read, the reinforcement reads, the reinforcement write, or service
boot -- ever hydrates, snapshots, or otherwise reads the whole graph.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
for _p in (_RECALL_ROOT, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import orion.substrate.falkor_direct as falkor_direct  # noqa: E402
from orion.substrate.falkor_direct import (  # noqa: E402
    CONCEPT_EDGE_CUT_CYPHER,
    CONCEPT_RANK_CYPHER,
    CONCEPT_ROWS_CYPHER,
    NODE_BY_ID_CYPHER,
    FalkorDirectConceptStore,
)
from orion.substrate.falkor_store import FalkorSubstrateStore  # noqa: E402
from orion.substrate.store import InMemorySubstrateGraphStore  # noqa: E402

from app import substrate_store  # noqa: E402
from app.collectors.concept_region import fetch_concept_region_fragment_and_reinforce  # noqa: E402

_OBSERVED = datetime(2026, 10, 6, tzinfo=timezone.utc).isoformat()


def _node_row(node_id: str, label: str, object_id: int, activation: float = 0.2) -> dict:
    return {
        "node_id": node_id, "node_kind": "concept", "identity_key": f"concept:{node_id}", "label": label,
        "definition": f"{label} def", "anchor_scope": "orion", "salience": 0.8, "confidence": 0.9,
        "activation": activation, "observed_at": _OBSERVED, "provenance_authority": "local_inferred",
        "provenance_source_kind": "test", "provenance_source_channel": "test", "provenance_producer": "t",
        "object_id": object_id,
    }


def _edge_row(edge_id: str, source: str, target: str, object_id: int) -> dict:
    return {
        "edge_id": edge_id, "identity_key": f"{source}|supports|{target}", "source_id": source,
        "source_kind": "concept", "target_id": target, "target_kind": "concept", "predicate": "supports",
        "substrate_edge": True, "salience": 0.7, "confidence": 0.6, "observed_at": _OBSERVED,
        "provenance_authority": "local_inferred", "provenance_source_kind": "test",
        "provenance_source_channel": "test", "provenance_producer": "t", "object_id": object_id,
    }


class _FakeFalkorClient:
    """Stands in for RedisGraphQueryClient. Answers only the bounded direct
    queries; any other Cypher (a hydration page scan, say) fails the test."""

    instances: list["_FakeFalkorClient"] = []

    def __init__(self, *, uri, graph_name, read_only=False, **timeouts):
        self.read_only = read_only
        self.timeouts = timeouts
        self.calls: list[tuple[str, dict]] = []
        _FakeFalkorClient.instances.append(self)

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        if self.read_only:
            if cypher == CONCEPT_RANK_CYPHER:
                return [{"object_id": 1, "label": "Juniper"}, {"object_id": 2, "label": "zebra"}]
            if cypher == CONCEPT_ROWS_CYPHER:
                return [_node_row("c-juniper", "Juniper", 1)] if params["object_ids"] == [1] else []
            if cypher == CONCEPT_EDGE_CUT_CYPHER:
                return [_edge_row("e1", "c-juniper", "c-zebra", 10)]
            if cypher == NODE_BY_ID_CYPHER:
                return [_node_row("c-juniper", "Juniper", 1)] if params["node_id"] == "c-juniper" else []
            raise AssertionError(f"unexpected read: {cypher[:80]}")
        if cypher.startswith("MERGE (n:SubstrateNode:"):
            return []
        if cypher == "CREATE INDEX FOR (n:SubstrateNode) ON (n.node_id)":
            return []
        raise AssertionError(f"unexpected write: {cypher[:80]}")


@pytest.fixture
def falkor_env(monkeypatch):
    _FakeFalkorClient.instances = []
    monkeypatch.setattr(falkor_direct, "RedisGraphQueryClient", _FakeFalkorClient)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")
    monkeypatch.setenv("FALKORDB_URI", "redis://falkor.test:6379")
    monkeypatch.setenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")
    whole_graph_reads: list[str] = []

    def _spy(name):
        def _record(self, *a, **k):
            whole_graph_reads.append(name)
            raise AssertionError(f"recall called {name}")
        return _record

    monkeypatch.setattr(FalkorSubstrateStore, "_hydrate_from_durable", _spy("FalkorSubstrateStore._hydrate_from_durable"))
    monkeypatch.setattr(FalkorSubstrateStore, "snapshot", _spy("FalkorSubstrateStore.snapshot"))
    monkeypatch.setattr(InMemorySubstrateGraphStore, "snapshot", _spy("InMemorySubstrateGraphStore.snapshot"))
    return whole_graph_reads


def test_recall_store_is_the_direct_falkor_handle_with_no_snapshot(falkor_env) -> None:
    store = substrate_store.get_substrate_store()
    assert isinstance(store._inner, FalkorDirectConceptStore)
    assert not hasattr(store, "snapshot")
    assert substrate_store.get_substrate_store() is store
    # Construction issued no graph read; the writer only ensured the node_id index.
    assert next(c for c in _FakeFalkorClient.instances if c.read_only).calls == []
    assert [cy for cy, _ in next(c for c in _FakeFalkorClient.instances if not c.read_only).calls] == [
        "CREATE INDEX FOR (n:SubstrateNode) ON (n.node_id)"
    ]
    # Reads are GRAPH.RO_QUERY; both clients carry the socket timeouts.
    assert sorted(c.read_only for c in _FakeFalkorClient.instances) == [False, True]
    assert all(
        c.timeouts == {
            "socket_timeout": substrate_store.FALKOR_SOCKET_TIMEOUT_S,
            "socket_connect_timeout": substrate_store.FALKOR_SOCKET_CONNECT_TIMEOUT_S,
        }
        for c in _FakeFalkorClient.instances
    )
    # A hung FalkorDB costs at most this per query (review finding 3: was 5s).
    assert substrate_store.FALKOR_SOCKET_TIMEOUT_S <= 1.5
    assert falkor_env == []


def test_concept_region_read_and_reinforce_never_hydrate(falkor_env) -> None:
    store = substrate_store.get_substrate_store()
    fragments = fetch_concept_region_fragment_and_reinforce("how is Juniper today", store=store)

    assert [f["id"] for f in fragments] == ["concept_region:node:c-juniper", "concept_region:edge:e1"]
    assert falkor_env == []
    reader = next(c for c in _FakeFalkorClient.instances if c.read_only)
    writer = next(c for c in _FakeFalkorClient.instances if not c.read_only)
    assert [cypher for cypher, _ in reader.calls] == [
        CONCEPT_RANK_CYPHER, CONCEPT_ROWS_CYPHER, CONCEPT_EDGE_CUT_CYPHER, NODE_BY_ID_CYPHER,
    ]  # one node read per reinforced node: node and identity key come from the same row
    merges = [(cy, p) for cy, p in writer.calls if cy.startswith("MERGE")]
    assert len(merges) == 1
    _cypher, params = merges[0]
    assert params["identity_key"] == "concept:c-juniper"
    assert params["activation"] == pytest.approx(0.2 + 0.8 * 0.08)


def test_no_label_match_is_a_single_light_query(falkor_env) -> None:
    store = substrate_store.get_substrate_store()
    assert fetch_concept_region_fragment_and_reinforce("nothing relevant here", store=store) == []
    reader = next(c for c in _FakeFalkorClient.instances if c.read_only)
    assert [cypher for cypher, _ in reader.calls] == [CONCEPT_RANK_CYPHER]
    assert falkor_env == []


def test_abandoned_call_reads_but_does_not_reinforce(falkor_env) -> None:
    store = substrate_store.get_substrate_store()
    abandoned = threading.Event()
    abandoned.set()
    fragments = fetch_concept_region_fragment_and_reinforce("Juniper", store=store, abandoned=abandoned)
    assert fragments
    assert all(
        not cy.startswith("MERGE") for c in _FakeFalkorClient.instances if not c.read_only for cy, _ in c.calls
    )
    assert falkor_env == []


def test_falkor_down_returns_empty_without_hydrating(falkor_env, monkeypatch) -> None:
    store = substrate_store.get_substrate_store()

    def _down(self, cypher, params=None):
        raise ConnectionError("falkor unreachable")

    monkeypatch.setattr(_FakeFalkorClient, "graph_query", _down)
    assert fetch_concept_region_fragment_and_reinforce("Juniper", store=store) == []
    assert falkor_env == []


def test_backends_that_bring_their_own_graph_cache_are_refused(monkeypatch, caplog) -> None:
    for backend in ("routed", "graphdb", "sparql"):
        substrate_store._reset_for_tests()
        monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", backend)
        with caplog.at_level(logging.WARNING, logger=substrate_store.logger.name):
            assert substrate_store.get_substrate_store() is None
        assert f"backend={backend}" in caplog.text
    substrate_store._reset_for_tests()
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "in_memory")
    assert isinstance(substrate_store.get_substrate_store(), InMemorySubstrateGraphStore)


def test_store_is_built_once_under_a_race(monkeypatch) -> None:
    builds: list = []
    monkeypatch.setattr(substrate_store, "_build_store", lambda: builds.append(1) or object())
    results: list = []
    threads = [threading.Thread(target=lambda: results.append(substrate_store.get_substrate_store())) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert builds == [1]
    assert len({id(r) for r in results}) == 1


# ── boot ────────────────────────────────────────────────────────────────────


class _FakeRabbit:
    def __init__(self, *a, **k):
        self.bus = object()
        self.handler = None

    async def start_background(self):
        return None

    async def stop(self):
        return None


def test_boot_does_not_touch_the_substrate_graph(monkeypatch) -> None:
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "Rabbit", _FakeRabbit)
    monkeypatch.setattr(main_mod, "chassis_cfg", lambda: None)
    monkeypatch.setattr(main_mod.settings, "RECALL_RDF_ENDPOINT_URL", "")
    monkeypatch.setattr(main_mod.settings, "RECALL_ENABLE_CARDS", False)
    monkeypatch.setattr(main_mod.settings, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(main_mod.settings, "RECALL_CONCEPT_REGION_ENABLED", True)
    builds: list = []
    monkeypatch.setattr(substrate_store, "_build_store", lambda: builds.append(1))

    async def _go():
        async with main_mod.lifespan(main_mod.app):
            return getattr(main_mod.app.state, "substrate_store_warmup", None)

    assert asyncio.run(_go()) is None
    assert builds == []
    assert not hasattr(substrate_store, "warm_substrate_store")


# ── review follow-ups (2026-10-06) ─────────────────────────────────────────


def _concept_node(node_id: str):
    from orion.substrate.falkor_codec import decode_node

    return decode_node(_node_row(node_id, node_id, 1))


class _RecordingStore:
    """Minimal store for reinforcement: combined read + recorded writes."""

    def __init__(self, node_ids, *, on_write=None):
        self.nodes = {nid: _concept_node(nid) for nid in node_ids}
        self.writes: list[str] = []
        self.reads: list[str] = []
        self._on_write = on_write

    def get_node_and_identity_key(self, node_id):
        self.reads.append(node_id)
        node = self.nodes.get(node_id)
        return node, (f"concept:{node_id}" if node else None)

    def get_node_by_id(self, node_id):  # pragma: no cover - must not be used
        raise AssertionError("separate read used")

    def get_identity_key_by_node_id(self, node_id):  # pragma: no cover
        raise AssertionError("separate read used")

    def upsert_node(self, *, identity_key, node, skip_metadata_keys=None):
        self.writes.append(node.node_id)
        if self._on_write:
            self._on_write(len(self.writes))


def test_reinforcement_reads_each_node_once() -> None:
    from app.collectors.concept_region import reinforce_matched_concepts

    store = _RecordingStore(["c1", "c2", "c3"])
    assert reinforce_matched_concepts(["c1", "c2", "c3"], store=store) == 3
    assert store.reads == ["c1", "c2", "c3"]
    assert store.writes == ["c1", "c2", "c3"]


def test_deadline_mid_reinforcement_loop_stops_further_writes() -> None:
    """Review finding 2: the abandoned check ran only before the loop, so a
    deadline that passed during node 1's write still let nodes 2..n write."""
    from app.collectors.concept_region import reinforce_matched_concepts

    abandoned = threading.Event()
    store = _RecordingStore(["c1", "c2", "c3", "c4"], on_write=lambda n: abandoned.set() if n == 1 else None)
    assert reinforce_matched_concepts(["c1", "c2", "c3", "c4"], store=store, abandoned=abandoned) == 1
    assert store.writes == ["c1"]
    assert store.reads == ["c1"]


def test_abandoned_is_forwarded_from_the_live_entry_point(monkeypatch) -> None:
    import app.collectors.concept_region as cr

    seen: list = []
    monkeypatch.setattr(
        cr, "fetch_concept_region_fragment",
        lambda q, *, store, limit_nodes, limit_edges: [{"id": f"{cr._NODE_FRAGMENT_ID_PREFIX}c1"}],
    )
    monkeypatch.setattr(cr, "reinforce_matched_concepts", lambda ids, *, store, abandoned=None: seen.append(abandoned))
    ev = threading.Event()
    cr.fetch_concept_region_fragment_and_reinforce("x", store=object(), abandoned=ev)
    assert seen == [ev]


class _TimeoutInner:
    def __init__(self):
        self.calls = 0

    def read_concept_region_matching(self, **_kw):
        from redis.exceptions import TimeoutError as RedisTimeoutError

        self.calls += 1
        raise RedisTimeoutError("Timeout reading from socket")


def test_breaker_opens_after_consecutive_timeouts_and_reopens_after_cooldown(caplog) -> None:
    clock = [100.0]
    breaker = substrate_store.ConceptRegionBreaker(threshold=3, cooldown_s=60.0, clock=lambda: clock[0])
    inner = _TimeoutInner()
    guarded = substrate_store._BreakerGuardedStore(inner, breaker)
    with caplog.at_level(logging.INFO, logger=substrate_store.logger.name):
        for _ in range(3):
            with pytest.raises(Exception):
                guarded.read_concept_region_matching(keep_label=None)
        assert breaker.is_open() and breaker.trips == 1
        # Open: refused without touching Falkor.
        with pytest.raises(substrate_store.ConceptRegionBreakerOpen):
            guarded.read_concept_region_matching(keep_label=None)
        assert inner.calls == 3
        clock[0] += 61.0
        assert not breaker.is_open()
        # Half-open probe times out -> reopens immediately.
        with pytest.raises(Exception):
            guarded.read_concept_region_matching(keep_label=None)
        assert breaker.is_open() and breaker.trips == 2
    assert "recall_concept_region_breaker_open" in caplog.text


def test_success_resets_the_consecutive_count() -> None:
    breaker = substrate_store.ConceptRegionBreaker(threshold=3, cooldown_s=60.0)
    breaker.record_timeout()
    breaker.record_timeout()
    breaker.record_success()
    breaker.record_timeout()
    assert not breaker.is_open()
    assert breaker.stats()["consecutive_timeouts"] == 1


def test_non_timeout_errors_do_not_trip_the_breaker() -> None:
    breaker = substrate_store.ConceptRegionBreaker(threshold=1, cooldown_s=60.0)

    class _Refused:
        def read_concept_region_matching(self, **_kw):
            raise ConnectionError("refused")

    guarded = substrate_store._BreakerGuardedStore(_Refused(), breaker)
    with pytest.raises(ConnectionError):
        guarded.read_concept_region_matching()
    assert not breaker.is_open()


def test_hung_falkor_is_bounded_per_turn_then_skipped(monkeypatch, caplog) -> None:
    """Review finding 3: a FalkorDB that accepts connections and never answers.
    Each turn is bounded by the socket read timeout; after the threshold the
    breaker skips concept_region entirely, with no connection attempt and no
    fallback."""
    import socket
    import time

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    port = server.getsockname()[1]
    accepted: list = []
    stop = threading.Event()

    def _accept_and_hang():
        server.settimeout(0.1)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            accepted.append(conn)  # never read, never answer

    th = threading.Thread(target=_accept_and_hang, daemon=True)
    th.start()
    try:
        monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")
        monkeypatch.setenv("FALKORDB_URI", f"redis://127.0.0.1:{port}")
        monkeypatch.setattr(substrate_store, "FALKOR_SOCKET_TIMEOUT_S", 0.3)
        monkeypatch.setattr(substrate_store, "FALKOR_SOCKET_CONNECT_TIMEOUT_S", 0.3)
        monkeypatch.setattr(substrate_store, "_BREAKER", substrate_store.ConceptRegionBreaker(threshold=3, cooldown_s=60.0))
        monkeypatch.setattr(FalkorSubstrateStore, "_hydrate_from_durable", lambda self: (_ for _ in ()).throw(AssertionError("hydrate")))

        elapsed = []
        for _ in range(3):
            started = time.perf_counter()
            assert fetch_concept_region_fragment_and_reinforce("Juniper", store=substrate_store.get_substrate_store()) == []
            elapsed.append(time.perf_counter() - started)
        assert all(e < 1.5 for e in elapsed), elapsed
        assert substrate_store.breaker_stats()["open"] is True
        assert substrate_store.breaker_stats()["timeouts"] == 3

        connections_before = len(accepted)
        with caplog.at_level(logging.INFO, logger=substrate_store.logger.name):
            started = time.perf_counter()
            assert substrate_store.get_substrate_store() is None
            assert fetch_concept_region_fragment_and_reinforce("Juniper", store=None) == []
            assert time.perf_counter() - started < 0.05
        assert len(accepted) == connections_before
        assert substrate_store.breaker_stats()["skipped"] == 1
        assert "recall_concept_region_breaker_skip" in caplog.text
    finally:
        stop.set()
        th.join(1.0)
        for conn in accepted:
            conn.close()
        server.close()
