"""FalkorAnchorStanceStore: the unification layer's store, without a full-graph hydrate.

Regression for 2026-10-06: every Hub turn's stance build paid one complete
Falkor hydrate (5,051 nodes / 38,393 edges, 14 s measured read-only; live
build_ms 17316-28632) because the 30 s refresh ceiling had always lapsed.

Unit lane (always runs): scripted client, proves one bounded read and no page
scan. Equivalence lane: needs ``ORION_TEST_FALKOR_URI`` pointing at a
throwaway FalkorDB (never production -- it writes and deletes its own graph).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    EvidenceNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate import falkor_anchor_store as anchor_mod
from orion.substrate.falkor_anchor_store import (
    ANCHOR_NODES_CYPHER,
    DEFAULT_STANCE_ANCHOR_SCOPES,
    FalkorAnchorStanceStore,
    build_unification_store_from_env,
)
from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig
from orion.substrate.relational.layer import CognitiveUnificationLayer
from orion.substrate.relational.registry import ProducerRegistryV1
from orion.substrate.store import InMemorySubstrateGraphStore

_NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)


def _prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(authority="local_inferred", source_kind="test", source_channel="test", producer="t")


def _signals(s: float = 0.5) -> SubstrateSignalBundleV1:
    return SubstrateSignalBundleV1(salience=s, confidence=0.5)


def _concept(node_id: str, anchor: str, s: float = 0.5) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id, label=f"label {node_id}", definition=f"def {node_id}", anchor_scope=anchor,
        temporal=SubstrateTemporalWindowV1(observed_at=_NOW), signals=_signals(s), provenance=_prov(),
    )


class _ScriptedClient:
    def __init__(self, responses=None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._responses = responses or {}
        self.read_only = True

    def graph_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        return list(self._responses.get(cypher, []))


class _NoHydrateWriter(FalkorSubstrateStore):
    def _hydrate_from_durable(self) -> None:  # pragma: no cover - must never run
        raise AssertionError("writer must never hydrate")


def _row(node_id: str, anchor: str, object_id: int) -> dict:
    return {
        "node_id": node_id, "node_kind": "concept", "identity_key": f"concept:{node_id}", "label": node_id,
        "anchor_scope": anchor, "observed_at": _NOW.isoformat(), "provenance_authority": "local_inferred",
        "provenance_source_kind": "test", "provenance_source_channel": "test", "provenance_producer": "t",
        "object_id": object_id,
    }


def _store(responses=None):
    read = _ScriptedClient(responses)
    write = _ScriptedClient()
    write.read_only = False
    writer = _NoHydrateWriter(FalkorSubstrateStoreConfig(uri="redis://unused"), client=write, hydrate=False)
    return FalkorAnchorStanceStore(read_client=read, writer=writer), read


def test_snapshot_is_one_bounded_read_with_no_page_scan() -> None:
    store, read = _store({ANCHOR_NODES_CYPHER: [_row("o1", "orion", 3), _row("j1", "juniper", 9)]})
    snap = store.snapshot()
    assert sorted(snap.nodes) == ["j1", "o1"]
    assert snap.edges == {}
    assert snap.node_identity_index == {"concept:o1": "o1", "concept:j1": "j1"}
    assert [c for c, _ in read.calls] == [ANCHOR_NODES_CYPHER]
    assert "world" not in read.calls[0][1]["anchors"]
    assert "LIMIT $page_size" not in ANCHOR_NODES_CYPHER
    assert store.snapshot_calls == 1 and store.snapshot_ms_total >= 0.0


def test_bad_row_is_skipped_not_fatal() -> None:
    bad = dict(_row("x", "orion", 1), node_kind="nonsense")
    store, _ = _store({ANCHOR_NODES_CYPHER: [bad, _row("o1", "orion", 2)]})
    assert list(store.snapshot().nodes) == ["o1"]


def test_world_scope_is_refused() -> None:
    read, write = _ScriptedClient(), _ScriptedClient()
    writer = _NoHydrateWriter(FalkorSubstrateStoreConfig(uri="redis://unused"), client=write, hydrate=False)
    with pytest.raises(ValueError):
        FalkorAnchorStanceStore(read_client=read, writer=writer, anchor_scopes=("orion", "world"))


def test_layer_refuses_an_anchor_the_store_does_not_snapshot() -> None:
    store, _ = _store()
    layer = CognitiveUnificationLayer(registry=ProducerRegistryV1(producers=[]), store=store)
    with pytest.raises(ValueError, match="world"):
        layer.beliefs_for_stance(anchors=("orion", "world"), ctx={})
    beliefs = layer.beliefs_for_stance(anchors=("orion", "juniper"), ctx={})
    assert set(beliefs.anchors) == {"orion", "juniper"}


def test_factory_selects_anchor_store_only_for_falkor(monkeypatch) -> None:
    monkeypatch.setattr(anchor_mod, "ensure_substrate_indexes", lambda *a, **k: None)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")
    monkeypatch.setenv("FALKORDB_URI", "redis://127.0.0.1:1")
    assert isinstance(build_unification_store_from_env(), FalkorAnchorStanceStore)
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "")
    assert isinstance(build_unification_store_from_env(), InMemorySubstrateGraphStore)


def test_cortex_exec_and_projection_layers_use_the_anchor_store(monkeypatch) -> None:
    """Both stance-path layer factories go through build_unification_store_from_env."""
    import inspect

    from orion.cognition import projection_builder

    assert "build_unification_store_from_env" in inspect.getsource(projection_builder.get_projection_unification_layer)
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "services", "orion-cortex-exec", "app", "chat_stance.py")
    src = open(os.path.abspath(path), encoding="utf-8").read()
    factory = src.split("def _get_unification_layer", 1)[1].split("\ndef ", 1)[0]
    assert "build_unification_store_from_env()" in factory


# ── equivalence lane: real throwaway FalkorDB ───────────────────────────────

_FALKOR_URI = os.getenv("ORION_TEST_FALKOR_URI", "").strip()
live = pytest.mark.skipif(not _FALKOR_URI, reason="ORION_TEST_FALKOR_URI not set (throwaway FalkorDB)")


@pytest.fixture(scope="module")
def mixed_graph():
    from orion.graph.falkor_client import RedisGraphQueryClient

    graph_name = f"test_falkor_anchor_{uuid.uuid4().hex[:10]}"
    client = RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name)
    writer = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=_FALKOR_URI, graph_name=graph_name), client=client, hydrate=False)
    refs = []
    for i, anchor in enumerate(["orion"] * 12 + ["juniper"] * 3 + ["relationship"] * 2 + ["claude"] + ["world"] * 30):
        node = _concept(f"n{i:03d}", anchor, s=(i % 7) / 7)
        writer.upsert_node(identity_key=f"concept:{node.node_id}", node=node)
        refs.append(NodeRefV1(node_id=node.node_id, node_kind="concept"))
    for i in range(40):
        ev = EvidenceNodeV1(
            node_id=f"ev{i:03d}", evidence_type="chat_turn", content_ref=f"turn:{i}", anchor_scope="world",
            temporal=SubstrateTemporalWindowV1(observed_at=_NOW), signals=_signals(1.0), provenance=_prov(),
        )
        writer.upsert_node(identity_key=f"evidence:{ev.node_id}", node=ev)
        refs.append(NodeRefV1(node_id=ev.node_id, node_kind="evidence"))
    for i in range(300):
        src, dst = refs[(i * 7) % len(refs)], refs[(i * 13 + 1) % len(refs)]
        edge = SubstrateEdgeV1(
            edge_id=f"e{i:04d}", source=src, target=dst, predicate="associated_with",
            temporal=SubstrateTemporalWindowV1(observed_at=_NOW), salience=(i % 5) / 5, confidence=0.5,
            provenance=_prov(),
        )
        writer.upsert_edge(identity_key=f"{src.node_id}|{dst.node_id}|{i}", edge=edge)
    yield graph_name
    client._r.execute_command("GRAPH.DELETE", graph_name)


def _stores(graph_name: str):
    from orion.graph.falkor_client import RedisGraphQueryClient

    cfg = FalkorSubstrateStoreConfig(uri=_FALKOR_URI, graph_name=graph_name)
    full = FalkorSubstrateStore(cfg)
    assert full.last_hydrate_ok is True
    anchor = FalkorAnchorStanceStore(
        read_client=RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name, read_only=True),
        writer=FalkorSubstrateStore(cfg, client=RedisGraphQueryClient(uri=_FALKOR_URI, graph_name=graph_name), hydrate=False),
    )
    return full, anchor


@live
def test_anchor_snapshot_equals_hydrated_snapshot_per_scope(mixed_graph) -> None:
    full, anchor = _stores(mixed_graph)
    full_nodes = full.snapshot().nodes
    got = anchor.snapshot().nodes
    for scope in DEFAULT_STANCE_ANCHOR_SCOPES:
        want = {k: v for k, v in full_nodes.items() if v.anchor_scope == scope}
        have = {k: v for k, v in got.items() if v.anchor_scope == scope}
        assert have == want, scope
    assert not any(v.anchor_scope == "world" for v in got.values())


@live
def test_concept_region_and_beliefs_match_the_hydrated_store(mixed_graph) -> None:
    from orion.substrate.relational.adapters.concept_induction_ctx import map_concept_induction_ctx_to_substrate
    from orion.substrate.relational.registry import CONCEPT_INDUCED_EPHEMERAL, ProducerEntryV1

    full, anchor = _stores(mixed_graph)
    a = anchor.query_concept_region(limit_nodes=64, limit_edges=64)
    f = full.query_concept_region(limit_nodes=64, limit_edges=64)
    assert a.slice.nodes == f.slice.nodes and a.slice.edges == f.slice.edges and not a.degraded

    def _layer(store):
        registry = ProducerRegistryV1(producers=[ProducerEntryV1(
            producer_id="concept_induction", trust_tier=CONCEPT_INDUCED_EPHEMERAL, anchor_scopes=("orion",),
            freshness_ttl_sec=0, pull_on_cold=True,
            adapter_fn=lambda ctx, store=store: map_concept_induction_ctx_to_substrate(ctx, store=store),
        )])
        return CognitiveUnificationLayer(registry=registry, store=store)

    anchors = ("orion", "relationship", "juniper")
    want = _layer(full).beliefs_for_stance(anchors=anchors, ctx={})
    have = _layer(anchor).beliefs_for_stance(anchors=anchors, ctx={})
    for name in anchors:
        assert sorted(n.node_id for n in have.anchors[name].concepts) == sorted(n.node_id for n in want.anchors[name].concepts)
    assert have.cold_anchors == want.cold_anchors and have.degraded_producers == want.degraded_producers


class _FailingAfterFirst(_ScriptedClient):
    def __init__(self, responses) -> None:
        super().__init__(responses)
        self.fail = False

    def graph_query(self, cypher, params=None):
        if self.fail:
            raise ConnectionError("falkor down")
        return super().graph_query(cypher, params)


def _store_with(read):
    write = _ScriptedClient()
    write.read_only = False
    writer = _NoHydrateWriter(FalkorSubstrateStoreConfig(uri="redis://unused"), client=write, hydrate=False)
    return FalkorAnchorStanceStore(read_client=read, writer=writer)


def test_read_error_serves_last_good_snapshot() -> None:
    read = _FailingAfterFirst({ANCHOR_NODES_CYPHER: [_row("o1", "orion", 1)]})
    store = _store_with(read)
    first = store.snapshot()
    read.fail = True
    assert store.snapshot() is first
    assert store.snapshot_failed_total == 1


def test_read_error_with_no_good_snapshot_raises() -> None:
    read = _FailingAfterFirst({})
    read.fail = True
    with pytest.raises(ConnectionError):
        _store_with(read).snapshot()


def test_duplicate_identity_resolves_to_nothing_like_the_lookup() -> None:
    a, b = _row("o1", "orion", 1), _row("o2", "orion", 2)
    b["identity_key"] = a["identity_key"]
    store, _ = _store({ANCHOR_NODES_CYPHER: [a, b]})
    snap = store.snapshot()
    assert sorted(snap.nodes) == ["o1", "o2"]
    assert a["identity_key"] not in snap.node_identity_index


def test_writer_cache_does_not_grow() -> None:
    store, _ = _store()
    store.upsert_node(identity_key="concept:o9", node=_concept("o9", "orion"))
    assert store._writer._cache.get_node_by_id("o9") is None
