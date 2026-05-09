"""Golden path integration scenario for the Cognitive Unification Layer.

Validates the full multi-producer → materialize → read-in-stance pipeline
using three producers: identity_yaml (operator_static), a stub autonomy
(graphdb_durable), and concept_induction (concept_induced).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    DriveNodeV1,
    StateSnapshotNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.store import InMemorySubstrateGraphStore
from orion.substrate.relational.layer import CognitiveUnificationLayer
from orion.substrate.relational.registry import (
    CONCEPT_INDUCED,
    GRAPHDB_DURABLE,
    OPERATOR_STATIC,
    ProducerEntryV1,
    ProducerRegistryV1,
)


# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------

def _make_prov(*, tier_rank: int) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="test",
        producer="test",
        tier_rank=tier_rank,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def identity_yaml_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Stub: one ConceptNodeV1 + one StateSnapshotNodeV1 at operator_static tier."""
    snapshot = StateSnapshotNodeV1(
        node_id="sub-identity-snapshot-orion",
        anchor_scope="orion",
        temporal=SubstrateTemporalWindowV1(observed_at=_now()),
        provenance=_make_prov(tier_rank=OPERATOR_STATIC.rank),
        signals=SubstrateSignalBundleV1(confidence=0.95, salience=0.8),
        snapshot_source="identity_yaml",
        dimensions={"identity_weight": 1.0},
        metadata={
            "orion_identity_summary": ["Orion is a cognitive presence."],
            "juniper_relationship_summary": ["Juniper is the co-architect."],
            "response_policy_summary": ["Answer directly."],
        },
        promotion_state="canonical",
    )
    concept = ConceptNodeV1(
        anchor_scope="orion",
        label="cognitive continuity",
        temporal=SubstrateTemporalWindowV1(observed_at=_now()),
        provenance=_make_prov(tier_rank=OPERATOR_STATIC.rank),
        signals=SubstrateSignalBundleV1(confidence=0.95, salience=0.8),
        metadata={"concept_id": "cognitive_continuity"},
        promotion_state="canonical",
    )
    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[snapshot, concept])


def autonomy_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Stub: one DriveNodeV1 at graphdb_durable tier."""
    drive = DriveNodeV1(
        anchor_scope="orion",
        drive_kind="growth",
        target_state="expanded_understanding",
        temporal=SubstrateTemporalWindowV1(observed_at=_now()),
        provenance=_make_prov(tier_rank=GRAPHDB_DURABLE.rank),
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.7),
    )
    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=[drive])


def concept_induction_adapter(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:
    """Stub: two ConceptNodeV1 at concept_induced tier."""
    nodes = [
        ConceptNodeV1(
            anchor_scope="orion",
            label=f"induced concept {i}",
            temporal=SubstrateTemporalWindowV1(observed_at=_now()),
            provenance=_make_prov(tier_rank=CONCEPT_INDUCED.rank),
            signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.4),
            metadata={"concept_id": f"induced_{i}"},
        )
        for i in range(2)
    ]
    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=nodes)


# ---------------------------------------------------------------------------
# Registry + Layer construction
# ---------------------------------------------------------------------------

def build_registry() -> ProducerRegistryV1:
    return ProducerRegistryV1(
        producers=[
            ProducerEntryV1(
                producer_id="identity_yaml",
                trust_tier=OPERATOR_STATIC,
                anchor_scopes=("orion",),
                freshness_ttl_sec=86400,
                pull_on_cold=True,
                adapter_fn=identity_yaml_adapter,
            ),
            ProducerEntryV1(
                producer_id="autonomy",
                trust_tier=GRAPHDB_DURABLE,
                anchor_scopes=("orion", "relationship", "juniper"),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=autonomy_adapter,
            ),
            ProducerEntryV1(
                producer_id="concept_induction",
                trust_tier=CONCEPT_INDUCED,
                anchor_scopes=("orion", "relationship", "juniper"),
                freshness_ttl_sec=300,
                pull_on_cold=True,
                adapter_fn=concept_induction_adapter,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGoldenPath:
    def test_cold_store_triggers_all_three_fan_outs(self):
        store = InMemorySubstrateGraphStore()
        layer = CognitiveUnificationLayer(registry=build_registry(), store=store)

        beliefs = layer.beliefs_for_stance(anchors=["orion"])

        assert "orion" in beliefs.cold_anchors, "Fresh store should be cold"
        assert len(beliefs.anchors["orion"].concepts) > 0, "Concepts from concept_induction"
        assert len(beliefs.anchors["orion"].drives) > 0, "Drives from autonomy"
        assert len(beliefs.anchors["orion"].snapshots) > 0, "Identity snapshot present"

    def test_identity_snapshot_has_operator_static_tier_rank(self):
        store = InMemorySubstrateGraphStore()
        layer = CognitiveUnificationLayer(registry=build_registry(), store=store)

        beliefs = layer.beliefs_for_stance(anchors=["orion"])

        snapshots = [n for n in beliefs.anchors["orion"].snapshots if getattr(n, "snapshot_source", "") == "identity_yaml"]
        assert snapshots, "Identity snapshot must be present"
        assert snapshots[0].provenance.tier_rank == OPERATOR_STATIC.rank

    def test_tier_protection_identity_confidence_not_lowered_by_concept_induced(self):
        """operator_static concept node confidence must be >= any concept_induced node's confidence."""
        store = InMemorySubstrateGraphStore()
        layer = CognitiveUnificationLayer(registry=build_registry(), store=store)

        beliefs = layer.beliefs_for_stance(anchors=["orion"])

        op_concepts = [
            n for n in beliefs.anchors["orion"].concepts
            if n.provenance.tier_rank == OPERATOR_STATIC.rank
        ]
        induced_concepts = [
            n for n in beliefs.anchors["orion"].concepts
            if n.provenance.tier_rank == CONCEPT_INDUCED.rank
        ]

        if op_concepts and induced_concepts:
            max_induced_confidence = max(n.signals.confidence for n in induced_concepts)
            for op_node in op_concepts:
                assert op_node.signals.confidence >= max_induced_confidence - 1e-6, (
                    "operator_static node confidence must not be overridden by concept_induced"
                )

    def test_cold_anchors_reported(self):
        store = InMemorySubstrateGraphStore()
        layer = CognitiveUnificationLayer(registry=build_registry(), store=store)

        beliefs = layer.beliefs_for_stance(anchors=["orion"])
        assert beliefs.cold_anchors == ["orion"]

    def test_warm_path_on_second_call(self):
        """Second call to beliefs_for_stance should not fan out (warm anchor)."""
        call_counts: dict[str, int] = {"identity_yaml": 0, "autonomy": 0, "concept_induction": 0}

        def _make_counting_adapter(original_fn, name):
            def _wrapped(ctx):
                call_counts[name] += 1
                return original_fn(ctx)
            return _wrapped

        registry = ProducerRegistryV1(
            producers=[
                ProducerEntryV1(
                    producer_id="identity_yaml",
                    trust_tier=OPERATOR_STATIC,
                    anchor_scopes=("orion",),
                    freshness_ttl_sec=86400,
                    pull_on_cold=True,
                    adapter_fn=_make_counting_adapter(identity_yaml_adapter, "identity_yaml"),
                ),
                ProducerEntryV1(
                    producer_id="autonomy",
                    trust_tier=GRAPHDB_DURABLE,
                    anchor_scopes=("orion",),
                    freshness_ttl_sec=300,
                    pull_on_cold=True,
                    adapter_fn=_make_counting_adapter(autonomy_adapter, "autonomy"),
                ),
                ProducerEntryV1(
                    producer_id="concept_induction",
                    trust_tier=CONCEPT_INDUCED,
                    anchor_scopes=("orion",),
                    freshness_ttl_sec=300,
                    pull_on_cold=True,
                    adapter_fn=_make_counting_adapter(concept_induction_adapter, "concept_induction"),
                ),
            ]
        )

        store = InMemorySubstrateGraphStore()
        layer = CognitiveUnificationLayer(registry=registry, store=store)

        # First call: cold → fan-out
        layer.beliefs_for_stance(anchors=["orion"])
        first_counts = dict(call_counts)

        # Second call: warm → no fan-out
        layer.beliefs_for_stance(anchors=["orion"])
        assert call_counts == first_counts, (
            "Second call (warm path) must not re-run any producer adapters"
        )

    def test_lineage_names_all_three_sources_with_tiers(self):
        store = InMemorySubstrateGraphStore()
        layer = CognitiveUnificationLayer(registry=build_registry(), store=store)

        beliefs = layer.beliefs_for_stance(anchors=["orion"])

        lineage_lower = " ".join(beliefs.lineage).lower()
        assert "identity_yaml" in lineage_lower
        assert "autonomy" in lineage_lower
        assert "concept_induction" in lineage_lower
        assert "operator_static" in lineage_lower
        assert "graphdb_durable" in lineage_lower
        assert "concept_induced" in lineage_lower

    def test_tier_outcomes_format_is_outcome_colon_count(self):
        """tier_outcomes strings must be in 'outcome_name:count' format, not raw lineage strings."""
        store = InMemorySubstrateGraphStore()
        layer = CognitiveUnificationLayer(registry=build_registry(), store=store)

        beliefs = layer.beliefs_for_stance(anchors=["orion"])

        # tier_outcomes may be empty if no merge decisions were recorded, but if present
        # every entry must match the "name:integer" format.
        for outcome_str in beliefs.anchors["orion"].tier_outcomes:
            parts = outcome_str.rsplit(":", 1)
            assert len(parts) == 2, f"tier_outcome must be 'name:count', got {outcome_str!r}"
            assert parts[1].isdigit(), f"tier_outcome count must be an integer, got {outcome_str!r}"
