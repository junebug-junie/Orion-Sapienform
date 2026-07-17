from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.routed_store import RoutedSubstrateGraphStore, build_routed_substrate_store_from_env
from orion.substrate.store import InMemorySubstrateGraphStore


def _concept(node_id: str = "sub-node-r1") -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        label="routed",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="test",
            producer="test_routed_store",
            evidence_refs=[],
        ),
    )


class _BoomStore(InMemorySubstrateGraphStore):
    def upsert_node(self, *, identity_key, node):
        raise RuntimeError("shadow boom")


def test_routed_writes_both_stores():
    primary = InMemorySubstrateGraphStore()
    shadow = InMemorySubstrateGraphStore()
    store = RoutedSubstrateGraphStore(primary=primary, shadow=shadow)
    node = _concept()
    store.upsert_node(identity_key="k1", node=node)
    assert primary.get_node_by_id(node.node_id) is not None
    assert shadow.get_node_by_id(node.node_id) is not None


def test_routed_shadow_failure_does_not_fail_primary():
    primary = InMemorySubstrateGraphStore()
    store = RoutedSubstrateGraphStore(primary=primary, shadow=_BoomStore())
    node = _concept()
    store.upsert_node(identity_key="k2", node=node)
    assert primary.get_node_by_id(node.node_id) is not None


def test_routed_reads_primary_only():
    primary = InMemorySubstrateGraphStore()
    shadow = InMemorySubstrateGraphStore()
    store = RoutedSubstrateGraphStore(primary=primary, shadow=shadow)
    only_shadow = _concept("shadow-only")
    shadow.upsert_node(identity_key="s", node=only_shadow)
    assert store.get_node_by_id("shadow-only") is None


def test_build_routed_from_env(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "routed")
    monkeypatch.setenv("SUBSTRATE_STORE_PRIMARY", "in_memory")
    monkeypatch.setenv("SUBSTRATE_STORE_SHADOW", "in_memory")
    from orion.substrate.graphdb_store import build_substrate_store_from_env

    store = build_substrate_store_from_env()
    assert isinstance(store, RoutedSubstrateGraphStore)
    node = _concept()
    store.upsert_node(identity_key="x", node=node)
    assert store.get_node_by_id(node.node_id) is not None
