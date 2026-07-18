from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from app.collectors.concept_region import (
    fetch_concept_region_fragment,
    fetch_concept_region_fragment_and_reinforce,
    reinforce_matched_concepts,
)

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    EvidenceNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
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


# --- reinforce_matched_concepts ---------------------------------------------


def test_reinforcement_bumps_activation_using_recall_boost_math():
    store = _seeded_store()
    before = store.snapshot().nodes["sub-concept-seed-juniper"]
    current = before.signals.activation.activation

    reinforced = reinforce_matched_concepts(["sub-concept-seed-juniper"], store=store)

    assert reinforced == 1
    after = store.snapshot().nodes["sub-concept-seed-juniper"]
    expected = min(1.0, current + (1.0 - current) * 0.08)
    assert after.signals.activation.activation == pytest.approx(expected, rel=1e-6)
    assert after.signals.activation.activation > current


def test_reinforcement_never_overshoots_ceiling():
    store = InMemorySubstrateGraphStore()
    node = ConceptNodeV1(
        node_id="sub-concept-hot",
        anchor_scope="orion",
        subject_ref="sub-concept-hot",
        promotion_state="canonical",
        temporal=_temporal(),
        provenance=_provenance(key="sub-concept-hot"),
        label="Hot Topic",
        definition="Definition of Hot Topic",
        signals=SubstrateSignalBundleV1(confidence=0.9, salience=1.0),
    )
    store.upsert_node(identity_key="concept|hot", node=node)
    assert store.snapshot().nodes["sub-concept-hot"].signals.activation.activation == 1.0

    reinforced = reinforce_matched_concepts(["sub-concept-hot"], store=store)

    assert reinforced == 1
    assert store.snapshot().nodes["sub-concept-hot"].signals.activation.activation == 1.0


def test_reinforcement_leaves_confidence_and_salience_untouched():
    store = _seeded_store()
    before = store.snapshot().nodes["sub-concept-seed-orion"]

    reinforce_matched_concepts(["sub-concept-seed-orion"], store=store)

    after = store.snapshot().nodes["sub-concept-seed-orion"]
    assert after.signals.confidence == before.signals.confidence
    assert after.signals.salience == before.signals.salience


def test_reinforcement_skips_non_concept_nodes():
    store = InMemorySubstrateGraphStore()
    evidence = EvidenceNodeV1(
        node_id="sub-evidence-1",
        anchor_scope="orion",
        temporal=_temporal(),
        provenance=_provenance(key="ev1"),
        evidence_type="test",
        content_ref="ref:1",
    )
    store.upsert_node(identity_key="evidence|1", node=evidence)

    reinforced = reinforce_matched_concepts(["sub-evidence-1"], store=store)

    assert reinforced == 0


def test_reinforcement_skips_unknown_node_ids():
    store = _seeded_store()

    reinforced = reinforce_matched_concepts(["sub-concept-does-not-exist"], store=store)

    assert reinforced == 0


def test_reinforcement_empty_ids_is_a_noop():
    store = _seeded_store()

    assert reinforce_matched_concepts([], store=store) == 0


def test_reinforcement_missing_store_degrades_to_zero_without_raising():
    assert reinforce_matched_concepts(["sub-concept-seed-orion"], store=None) == 0


def test_reinforcement_raising_store_degrades_to_zero_without_raising():
    class _RaisingStore:
        def get_node_by_id(self, node_id):
            raise RuntimeError("boom: store unavailable")

    assert reinforce_matched_concepts(["sub-concept-seed-orion"], store=_RaisingStore()) == 0


def test_reinforcement_missing_identity_key_skips_rather_than_clobbers():
    # Regression: FalkorSubstrateStore's codec writes `identity_key or ""`
    # unconditionally on every upsert. A node present in the cache but
    # missing from the identity index must never be reinforced with a
    # falsy identity_key -- that would durably wipe its real identity on a
    # real Falkor backend. get_identity_key_by_node_id() returning None
    # (e.g. this node was upserted with identity_key=None) must skip, not
    # write.
    store = InMemorySubstrateGraphStore()
    node = _concept_node(node_id="sub-concept-no-identity", label="No Identity")
    store.upsert_node(identity_key=None, node=node)

    reinforced = reinforce_matched_concepts(["sub-concept-no-identity"], store=store)

    assert reinforced == 0
    assert store.get_identity_key_by_node_id("sub-concept-no-identity") is None


# --- fetch_concept_region_fragment_and_reinforce ----------------------------


def test_fetch_and_reinforce_bumps_only_matched_concepts():
    store = _seeded_store()
    before_juniper = store.snapshot().nodes["sub-concept-seed-juniper"].signals.activation.activation
    before_orion = store.snapshot().nodes["sub-concept-seed-orion"].signals.activation.activation

    fragments = fetch_concept_region_fragment_and_reinforce(_Query("tell me about Juniper today"), store=store)

    assert fragments
    after_juniper = store.snapshot().nodes["sub-concept-seed-juniper"].signals.activation.activation
    after_orion = store.snapshot().nodes["sub-concept-seed-orion"].signals.activation.activation
    assert after_juniper > before_juniper
    assert after_orion == before_orion  # Orion wasn't mentioned, wasn't matched, untouched


def test_fetch_and_reinforce_returns_same_fragments_as_plain_fetch():
    store_a = _seeded_store()
    store_b = _seeded_store()

    plain = fetch_concept_region_fragment(_Query("tell me about Juniper today"), store=store_a)
    combined = fetch_concept_region_fragment_and_reinforce(_Query("tell me about Juniper today"), store=store_b)

    assert [f["id"] for f in plain] == [f["id"] for f in combined]


def test_fetch_and_reinforce_no_match_does_not_write():
    store = _seeded_store()
    before = store.snapshot().nodes["sub-concept-seed-juniper"].signals.activation.activation

    fragments = fetch_concept_region_fragment_and_reinforce(_Query("what's the weather like"), store=store)

    assert fragments == []
    after = store.snapshot().nodes["sub-concept-seed-juniper"].signals.activation.activation
    assert after == before


def test_fetch_and_reinforce_missing_store_degrades_to_empty_without_raising():
    fragments = fetch_concept_region_fragment_and_reinforce(_Query("mentions Juniper"), store=None)

    assert fragments == []
