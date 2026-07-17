"""Tests for concept_induction_ctx adapter.

Covers: a store with real concept nodes produces a non-None
SubstrateGraphRecordV1 with nodes correctly bucketed/tagged; an empty store
degrades to None; a store connection failure degrades to None without
raising; the tier_rank stamping still happens correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
)
from orion.substrate.adapters._common import make_temporal
from orion.substrate.relational.adapters import concept_induction_ctx as module
from orion.substrate.store import InMemorySubstrateGraphStore


@pytest.fixture(autouse=True)
def _reset_store_singleton():
    """The adapter caches a process-level store singleton; isolate tests."""
    module._STORE = None
    yield
    module._STORE = None


def _make_concept(
    *,
    node_id: str,
    anchor_scope: str,
    label: str,
    concept_type: str | None = None,
    node_kind: str = "concept",
) -> ConceptNodeV1:
    node = ConceptNodeV1(
        node_id=node_id,
        anchor_scope=anchor_scope,
        subject_ref=f"entity:{anchor_scope}",
        temporal=make_temporal(observed_at=datetime.now(timezone.utc)),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="seed_concept",
            source_channel="substrate.seed",
            producer="test_fixture",
        ),
        label=label,
        metadata={"concept_type": concept_type} if concept_type else {},
    )
    assert node.node_kind == node_kind
    return node


def _populated_store() -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    store.upsert_node(
        identity_key="concept|orion|self-continuity",
        node=_make_concept(node_id="c-orion-1", anchor_scope="orion", label="self:continuity"),
    )
    store.upsert_node(
        identity_key="concept|relationship|trust",
        node=_make_concept(node_id="c-rel-1", anchor_scope="relationship", label="relationship:trust"),
    )
    store.upsert_node(
        identity_key="concept|juniper|architect",
        node=_make_concept(node_id="c-juniper-1", anchor_scope="juniper", label="juniper:co_architect"),
    )
    # A concept outside the registered anchor_scopes must be filtered out.
    store.upsert_node(
        identity_key="concept|world|weather",
        node=_make_concept(node_id="c-world-1", anchor_scope="world", label="world:weather"),
    )
    # An explicit concept_type must be preserved, not overwritten.
    store.upsert_node(
        identity_key="concept|orion|tension",
        node=_make_concept(
            node_id="c-orion-2",
            anchor_scope="orion",
            label="tension:autonomy_vs_safety",
            concept_type="tension",
        ),
    )
    return store


def test_populated_store_produces_record_with_bucketed_nodes():
    module._STORE = _populated_store()

    record = module.map_concept_induction_ctx_to_substrate({})

    assert record is not None
    assert record.anchor_scope == "orion"
    node_ids = {n.node_id for n in record.nodes}
    # world-scoped node filtered out (not in orion/relationship/juniper).
    assert node_ids == {"c-orion-1", "c-rel-1", "c-juniper-1", "c-orion-2"}

    by_id = {n.node_id: n for n in record.nodes}
    # anchor_scope="orion" defaults to concept_type="self".
    assert by_id["c-orion-1"].metadata["concept_type"] == "self"
    # anchor_scope="relationship" defaults to concept_type="relationship".
    assert by_id["c-rel-1"].metadata["concept_type"] == "relationship"
    # anchor_scope="juniper" defaults to concept_type="self" (mirrors the
    # chat_stance.py::_concept_summary_from_store subject-bucketing precedent).
    assert by_id["c-juniper-1"].metadata["concept_type"] == "self"
    # Pre-set concept_type is preserved, not clobbered by the anchor_scope default.
    assert by_id["c-orion-2"].metadata["concept_type"] == "tension"


def test_tier_rank_stamped_on_every_node():
    module._STORE = _populated_store()

    record = module.map_concept_induction_ctx_to_substrate({})

    assert record is not None
    assert record.nodes
    for node in record.nodes:
        assert node.provenance.tier_rank == module._TIER_RANK == 3


def test_empty_store_degrades_to_none():
    module._STORE = InMemorySubstrateGraphStore()

    record = module.map_concept_induction_ctx_to_substrate({})

    assert record is None


def test_store_with_only_out_of_scope_anchors_degrades_to_none():
    store = InMemorySubstrateGraphStore()
    store.upsert_node(
        identity_key="concept|world|weather",
        node=_make_concept(node_id="c-world-1", anchor_scope="world", label="world:weather"),
    )
    module._STORE = store

    record = module.map_concept_induction_ctx_to_substrate({})

    assert record is None


def test_store_construction_failure_degrades_to_none(monkeypatch):
    def _boom():
        raise ConnectionError("falkor unreachable")

    monkeypatch.setattr(module, "build_substrate_store_from_env", _boom)

    record = module.map_concept_induction_ctx_to_substrate({})

    assert record is None


def test_store_query_failure_degrades_to_none():
    class _BoomStore:
        def query_concept_region(self, *, limit_nodes=64, limit_edges=64):
            raise RuntimeError("falkor connection dropped mid-query")

    module._STORE = _BoomStore()

    record = module.map_concept_induction_ctx_to_substrate({})

    assert record is None


def test_degraded_query_result_degrades_to_none():
    from orion.substrate.store import SubstrateNeighborhoodSliceV1, SubstrateQueryResultV1

    class _DegradedStore:
        def query_concept_region(self, *, limit_nodes=64, limit_edges=64):
            return SubstrateQueryResultV1(
                query_kind="concept_region",
                slice=SubstrateNeighborhoodSliceV1(nodes=[], edges=[]),
                source_kind="falkor",
                degraded=True,
                error="connection reset",
            )

    module._STORE = _DegradedStore()

    record = module.map_concept_induction_ctx_to_substrate({})

    assert record is None


def test_none_ctx_does_not_raise():
    module._STORE = _populated_store()

    record = module.map_concept_induction_ctx_to_substrate(None)

    assert record is not None
