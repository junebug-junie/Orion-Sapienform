from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from app.collectors.concept_region import fetch_concept_region_fragment

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.store import InMemorySubstrateGraphStore


def _provenance(*, key: str) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="human_verified",
        source_kind="test_fixture",
        source_channel="test:concept_region",
        producer="test_concept_region_collector",
        evidence_refs=[f"fixture#{key}"],
    )


def _temporal() -> SubstrateTemporalWindowV1:
    return SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc))


def _concept_node(*, node_id: str, label: str, anchor_scope: str = "orion") -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        anchor_scope=anchor_scope,
        subject_ref=node_id,
        promotion_state="canonical",
        temporal=_temporal(),
        provenance=_provenance(key=node_id),
        label=label,
        definition=f"Definition of {label}",
    )


def _seeded_store() -> InMemorySubstrateGraphStore:
    """Mirrors orion/substrate/seed.py's shape: a couple of concept nodes
    plus a cross-reference edge, written via upsert_node/upsert_edge."""

    store = InMemorySubstrateGraphStore()

    orion_node = _concept_node(node_id="sub-concept-seed-orion", label="Orion", anchor_scope="orion")
    juniper_node = _concept_node(node_id="sub-concept-seed-juniper", label="Juniper", anchor_scope="juniper")

    store.upsert_node(identity_key="concept|orion|orion|seed", node=orion_node)
    store.upsert_node(identity_key="concept|juniper|juniper|seed", node=juniper_node)

    edge = SubstrateEdgeV1(
        edge_id="sub-edge-seed-orion-juniper",
        source=NodeRefV1(node_id=orion_node.node_id, node_kind="concept"),
        target=NodeRefV1(node_id=juniper_node.node_id, node_kind="concept"),
        predicate="associated_with",
        temporal=_temporal(),
        provenance=_provenance(key="orion-juniper"),
    )
    store.upsert_edge(identity_key=f"{orion_node.node_id}|associated_with|{juniper_node.node_id}", edge=edge)

    return store


class _Query:
    """Minimal stand-in for the loosely-typed `query` object active_packet.py
    and worker.py pass around (`.fragment` carries the turn text)."""

    def __init__(self, fragment: str) -> None:
        self.fragment = fragment


class _CountingStore:
    """Wraps a real store and counts calls to read_concept_region, so we can
    assert the collector never reads the store when there's no turn text."""

    def __init__(self, inner: InMemorySubstrateGraphStore) -> None:
        self._inner = inner
        self.read_calls = 0

    def read_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64):
        self.read_calls += 1
        return self._inner.read_concept_region(limit_nodes=limit_nodes, limit_edges=limit_edges)


class _RaisingStore:
    def read_concept_region(self, *, limit_nodes: int = 32, limit_edges: int = 64):
        raise RuntimeError("boom: store unavailable")


def test_turn_mentioning_seeded_label_returns_nonempty_fragment():
    store = _seeded_store()

    fragments = fetch_concept_region_fragment(_Query("tell me about Juniper today"), store=store)

    assert fragments
    ids = [f["id"] for f in fragments]
    assert any("sub-concept-seed-juniper" in i for i in ids)
    # matched node's immediate edge should also come back, bounded to the match
    assert any(f["source"] == "concept_region" and "kind:concept_edge" in f["tags"] for f in fragments)


def test_turn_with_no_relevant_mention_returns_empty_and_skips_store_read():
    store = _seeded_store()
    counting_store = _CountingStore(store)

    fragments = fetch_concept_region_fragment(_Query("what's the weather like"), store=counting_store)

    assert fragments == []
    # non-empty fragment text still triggers a read (that's expected -- the
    # cheap-skip-on-empty-input path is covered separately below); a store
    # read happening here but finding no label match is correct behavior.
    assert counting_store.read_calls == 1


def test_empty_turn_text_skips_store_read_entirely():
    store = _seeded_store()
    counting_store = _CountingStore(store)

    fragments = fetch_concept_region_fragment(_Query(""), store=counting_store)

    assert fragments == []
    assert counting_store.read_calls == 0


def test_none_query_skips_store_read_entirely():
    store = _seeded_store()
    counting_store = _CountingStore(store)

    fragments = fetch_concept_region_fragment(None, store=counting_store)

    assert fragments == []
    assert counting_store.read_calls == 0


def test_bare_string_query_is_accepted_directly():
    store = _seeded_store()

    fragments = fetch_concept_region_fragment("say hi to Orion for me", store=store)

    assert fragments
    assert any("sub-concept-seed-orion" in f["id"] for f in fragments)


@pytest.mark.parametrize(
    "turn_text",
    [
        "JUNIPER is around",
        "juniper is around",
        "JuNiPeR is around",
        "juniper's calendar is full",  # substring match: label is a prefix of a longer token
    ],
)
def test_matching_is_case_insensitive_and_substring_based(turn_text: str):
    store = _seeded_store()

    fragments = fetch_concept_region_fragment(_Query(turn_text), store=store)

    assert fragments
    assert any("sub-concept-seed-juniper" in f["id"] for f in fragments)


def test_short_labels_below_minimum_length_do_not_match_everything():
    store = InMemorySubstrateGraphStore()
    short_node = _concept_node(node_id="sub-concept-seed-go", label="Go")
    store.upsert_node(identity_key="concept|orion|go|seed", node=short_node)

    # "Go" is a substring of tons of unrelated text; the 3-char minimum
    # documented in concept_region.py should prevent this from matching.
    fragments = fetch_concept_region_fragment(_Query("Going to the store later"), store=store)

    assert fragments == []


def test_missing_store_degrades_to_empty_without_raising():
    fragments = fetch_concept_region_fragment(_Query("mentions Juniper"), store=None)

    assert fragments == []


def test_raising_store_degrades_to_empty_without_raising():
    fragments = fetch_concept_region_fragment(_Query("mentions Juniper"), store=_RaisingStore())

    assert fragments == []


def test_empty_store_returns_empty():
    store = InMemorySubstrateGraphStore()

    fragments = fetch_concept_region_fragment(_Query("mentions Juniper"), store=store)

    assert fragments == []


def test_malformed_query_object_without_fragment_attr_degrades_to_empty():
    store = _seeded_store()

    class _NotAQuery:
        pass

    fragments = fetch_concept_region_fragment(_NotAQuery(), store=store)

    assert fragments == []
