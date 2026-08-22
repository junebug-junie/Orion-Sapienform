from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from orion.core.schemas.concept_induction import ConceptCluster, ConceptEvidenceRef, ConceptItem, ConceptProfile, ConceptProfileDelta
from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
from orion.core.schemas.spark_canonical import SparkSourceSnapshotV1
from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.substrate.adapters import (
    map_concept_delta_to_substrate,
    map_concept_profile_to_substrate,
    map_spark_source_snapshot_to_substrate,
    map_spark_state_snapshot_to_substrate,
)


def _concept_profile() -> ConceptProfile:
    now = datetime.now(timezone.utc)
    evidence = ConceptEvidenceRef(message_id=uuid4(), timestamp=now, channel="orion:test")
    return ConceptProfile(
        subject="orion",
        window_start=now - timedelta(hours=1),
        window_end=now,
        concepts=[
            ConceptItem(
                concept_id="c1",
                label="coherence",
                type="motif",
                confidence=0.8,
                salience=0.7,
                evidence=[evidence],
            )
        ],
        clusters=[ConceptCluster(cluster_id="cluster-1", label="cog", summary="coherence cluster", concept_ids=["c1"], cohesion_score=0.72)],
        metadata={"subject_ref": "project:orion_sapienform"},
    )


def test_concept_adapter_maps_profile_and_preserves_scope_subject_and_provenance() -> None:
    profile = _concept_profile()
    out = map_concept_profile_to_substrate(profile=profile, anchor_scope="orion")
    assert isinstance(out, SubstrateGraphRecordV1)
    assert out.subject_ref == "project:orion_sapienform"
    concept_nodes = [n for n in out.nodes if n.node_kind == "concept"]
    evidence_nodes = [n for n in out.nodes if n.node_kind == "evidence"]
    hypothesis_nodes = [n for n in out.nodes if n.node_kind == "hypothesis"]
    assert len(concept_nodes) == 1
    assert len(evidence_nodes) >= 1
    assert len(hypothesis_nodes) == 1  # conservative cluster mapping
    assert any(edge.predicate == "supports" for edge in out.edges)
    assert concept_nodes[0].provenance.source_kind == "concept_induction.profile"


def test_concept_adapter_links_induced_concept_to_its_golden_subject_anchor() -> None:
    # The fix for real induced concepts being permanently zero-degree
    # (2026-08-22): a concept induced from subject "orion"'s window gets a
    # real associated_with edge from Orion's own golden seed node
    # (orion/substrate/seed_concepts.yaml), not a fabricated relationship --
    # ConceptProfile.subject is a structural fact already carried by the
    # profile, not an inference.
    profile = _concept_profile()  # subject="orion"
    out = map_concept_profile_to_substrate(profile=profile, anchor_scope="orion")
    concept_node_id = next(n.node_id for n in out.nodes if n.node_kind == "concept")
    anchor_edges = [
        e
        for e in out.edges
        if e.source.node_id == "sub-concept-seed-orion" and e.target.node_id == concept_node_id
    ]
    assert len(anchor_edges) == 1
    assert anchor_edges[0].predicate == "associated_with"
    assert anchor_edges[0].source.node_kind == "concept"
    assert anchor_edges[0].target.node_kind == "concept"


def test_concept_adapter_skips_subject_anchor_edge_for_unrecognized_subject() -> None:
    # ConceptProfile.subject is a plain str (not a schema-enforced Literal --
    # see its own field description). An unrecognized value must degrade to
    # no anchor edge, never a fabricated reference to a node we can't
    # confirm exists.
    now = datetime.now(timezone.utc)
    evidence = ConceptEvidenceRef(message_id=uuid4(), timestamp=now, channel="orion:test")
    profile = ConceptProfile(
        subject="some_unknown_subject",
        window_start=now - timedelta(hours=1),
        window_end=now,
        concepts=[
            ConceptItem(
                concept_id="c-unknown",
                label="orphan",
                type="motif",
                confidence=0.5,
                salience=0.5,
                evidence=[evidence],
            )
        ],
    )
    out = map_concept_profile_to_substrate(profile=profile, anchor_scope="world")
    assert not any(
        e.predicate == "associated_with" and e.source.node_id.startswith("sub-concept-seed-") for e in out.edges
    )


def test_concept_adapter_seeds_activation_and_half_life_from_salience() -> None:
    # Regression: signals.activation used to be left at the schema default
    # (activation=0.0, decay_half_life_seconds=None), making Hub's decay
    # scheduler a permanent no-op for every concept node it touched.
    profile = _concept_profile()
    out = map_concept_profile_to_substrate(profile=profile, anchor_scope="orion")
    concept_nodes = [n for n in out.nodes if n.node_kind == "concept"]
    assert concept_nodes
    node = concept_nodes[0]
    assert node.signals.activation.activation == node.signals.salience
    assert node.signals.activation.activation > 0.0
    assert node.signals.activation.decay_half_life_seconds is not None
    assert node.signals.activation.decay_half_life_seconds > 0


def test_concept_delta_adapter_only_emits_contradiction_when_semantics_support_it() -> None:
    now = datetime.now(timezone.utc)
    delta = ConceptProfileDelta(profile_id="p1", from_rev=1, to_rev=2, added=["c2"], removed=["c1"], rationale="concept conflict")
    out = map_concept_delta_to_substrate(delta=delta, observed_at=now, anchor_scope="orion", subject_ref="project:orion")
    contradiction_nodes = [n for n in out.nodes if n.node_kind == "contradiction"]
    assert len(contradiction_nodes) == 1
    assert all(edge.predicate == "contradicts" for edge in out.edges)


def test_spark_adapter_maps_snapshots_conservatively() -> None:
    now = datetime.now(timezone.utc)
    source_snapshot = SparkSourceSnapshotV1(
        source_service="orion:spark",
        snapshot_ts=now,
        source_snapshot_id="snap-1",
        dimensions={"focus": 0.4},
        tensions=["novelty_pressure"],
        metadata={"subject_ref": "project:orion_sapienform"},
    )
    source_out = map_spark_source_snapshot_to_substrate(snapshot=source_snapshot)
    assert any(n.node_kind == "state_snapshot" for n in source_out.nodes)
    assert any(n.node_kind == "tension" for n in source_out.nodes)

    state_snapshot = SparkStateSnapshotV1(
        source_service="orion:spark",
        producer_boot_id="boot-1",
        seq=7,
        snapshot_ts=now,
        phi={"coherence": 0.6},
        valence=0.6,
        arousal=0.4,
        dominance=0.5,
        metadata={"subject_ref": "project:orion_sapienform", "transition_event": "focus_shift"},
    )
    state_out = map_spark_state_snapshot_to_substrate(snapshot=state_snapshot)
    assert any(n.node_kind == "state_snapshot" for n in state_out.nodes)
    assert any(n.node_kind == "event" for n in state_out.nodes)
    assert all(n.node_kind != "entity" for n in state_out.nodes)


def test_non_destructive_existing_domain_schemas_remain_usable() -> None:
    profile = _concept_profile()
    assert profile.concepts[0].label == "coherence"
    spark_state = SparkStateSnapshotV1(
        source_service="orion:spark",
        producer_boot_id="boot-2",
        seq=1,
        snapshot_ts=datetime.now(timezone.utc),
    )
    assert spark_state.idempotency_key == "boot-2:1"
