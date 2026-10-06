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
    assert isinstance(store, FalkorDirectConceptStore)
    assert not hasattr(store, "snapshot")
    assert substrate_store.get_substrate_store() is store
    # Construction issued no query at all.
    assert all(client.calls == [] for client in _FakeFalkorClient.instances)
    # Reads are GRAPH.RO_QUERY; both clients carry the socket timeouts.
    assert sorted(c.read_only for c in _FakeFalkorClient.instances) == [False, True]
    assert all(
        c.timeouts == {
            "socket_timeout": substrate_store.FALKOR_SOCKET_TIMEOUT_S,
            "socket_connect_timeout": substrate_store.FALKOR_SOCKET_CONNECT_TIMEOUT_S,
        }
        for c in _FakeFalkorClient.instances
    )
    assert falkor_env == []


def test_concept_region_read_and_reinforce_never_hydrate(falkor_env) -> None:
    store = substrate_store.get_substrate_store()
    fragments = fetch_concept_region_fragment_and_reinforce("how is Juniper today", store=store)

    assert [f["id"] for f in fragments] == ["concept_region:node:c-juniper", "concept_region:edge:e1"]
    assert falkor_env == []
    reader = next(c for c in _FakeFalkorClient.instances if c.read_only)
    writer = next(c for c in _FakeFalkorClient.instances if not c.read_only)
    assert [cypher for cypher, _ in reader.calls] == [
        CONCEPT_RANK_CYPHER, CONCEPT_ROWS_CYPHER, CONCEPT_EDGE_CUT_CYPHER, NODE_BY_ID_CYPHER, NODE_BY_ID_CYPHER,
    ]
    assert len(writer.calls) == 1
    _cypher, params = writer.calls[0]
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
    assert all(c.calls == [] for c in _FakeFalkorClient.instances if not c.read_only)
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
