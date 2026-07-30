from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from orion.core.schemas.concept_induction import ConceptEvidenceRef, ConceptItem, ConceptProfile
from orion.core.schemas.spark_canonical import SparkSourceSnapshotV1
from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.substrate import (
    InMemorySubstrateGraphStore,
    SubstrateGraphMaterializer,
    map_concept_profile_to_substrate,
    map_spark_source_snapshot_to_substrate,
    map_spark_state_snapshot_to_substrate,
)


def _concept_profile(*, concept_id: str, label: str) -> ConceptProfile:
    now = datetime.now(timezone.utc)
    evidence = ConceptEvidenceRef(message_id=uuid4(), timestamp=now, channel="orion:test")
    return ConceptProfile(
        profile_id=f"profile-{concept_id}",
        subject="orion",
        window_start=now - timedelta(hours=1),
        window_end=now,
        concepts=[ConceptItem(concept_id=concept_id, label=label, confidence=0.8, salience=0.6, evidence=[evidence])],
        metadata={"subject_ref": "project:orion"},
    )


def test_node_identity_reconciles_concepts_conservatively() -> None:
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)

    rec1 = map_concept_profile_to_substrate(profile=_concept_profile(concept_id="c1", label="Coherence"))
    rec2 = map_concept_profile_to_substrate(profile=_concept_profile(concept_id="c1", label="Coherence"))
    res1 = materializer.apply_record(rec1)
    res2 = materializer.apply_record(rec2)
    assert res1.nodes_created >= 1
    assert res2.nodes_merged >= 1


def test_edge_reconciliation_is_deterministic_and_avoids_duplicate_spam() -> None:
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)

    profile = _concept_profile(concept_id="c2", label="Continuity")
    record = map_concept_profile_to_substrate(profile=profile)
    materializer.apply_record(record)
    materializer.apply_record(record)
    snapshot = store.snapshot()

    supports_edges = [edge for edge in snapshot.edges.values() if edge.predicate == "supports"]
    assert len(supports_edges) == 1
    assert supports_edges[0].metadata.get("materialization_lineage")


def test_provenance_and_lineage_are_preserved_on_merge() -> None:
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)

    first = map_concept_profile_to_substrate(profile=_concept_profile(concept_id="c3", label="Novelty"))
    second = map_concept_profile_to_substrate(profile=_concept_profile(concept_id="c3", label="Novelty"))
    materializer.apply_record(first)
    materializer.apply_record(second)

    nodes = [node for node in store.snapshot().nodes.values() if node.node_kind == "concept"]
    assert len(nodes) == 1
    concept = nodes[0]
    lineage = concept.metadata.get("materialization_lineage") or []
    assert len(lineage) >= 1
    assert concept.provenance.evidence_refs


def test_materialized_store_persists_state_across_repeated_application() -> None:
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)

    spark_source = SparkSourceSnapshotV1(
        source_service="orion:spark",
        snapshot_ts=datetime.now(timezone.utc),
        source_snapshot_id="src-1",
        dimensions={"focus": 0.5},
        tensions=["novelty_pressure"],
    )
    spark_state = SparkStateSnapshotV1(
        source_service="orion:spark",
        producer_boot_id="boot-1",
        seq=1,
        snapshot_ts=datetime.now(timezone.utc),
        metadata={"transition_event": "focus_shift"},
    )
    recs = [
        map_spark_source_snapshot_to_substrate(snapshot=spark_source),
        map_spark_state_snapshot_to_substrate(snapshot=spark_state),
    ]
    results = materializer.apply_records(recs)
    assert len(results) == 2
    assert store.snapshot().nodes
    assert store.snapshot().edges


def test_phase2_adapters_remain_valid_inputs_non_destructive() -> None:
    materializer = SubstrateGraphMaterializer()
    concept_record = map_concept_profile_to_substrate(profile=_concept_profile(concept_id="c4", label="Coherence"))
    spark_record = map_spark_source_snapshot_to_substrate(
        snapshot=SparkSourceSnapshotV1(
            source_service="orion:spark",
            snapshot_ts=datetime.now(timezone.utc),
            source_snapshot_id="src-c4",
            dimensions={"focus": 0.5},
            tensions=["novelty_pressure"],
        )
    )
    out = materializer.apply_records([concept_record, spark_record])
    assert len(out) == 2
    assert all(item.nodes_seen >= 1 for item in out)
