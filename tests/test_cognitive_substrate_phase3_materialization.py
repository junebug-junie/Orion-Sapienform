from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from orion.core.schemas.concept_induction import ConceptEvidenceRef, ConceptItem, ConceptProfile
from orion.core.schemas.drives import ArtifactProvenance, DriveStateV1, GoalProposalV1
from orion.core.schemas.spark_canonical import SparkSourceSnapshotV1
from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.substrate import (
    InMemorySubstrateGraphStore,
    SubstrateGraphMaterializer,
    map_autonomy_artifacts_to_substrate,
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


def _provenance() -> ArtifactProvenance:
    return ArtifactProvenance(intake_channel="orion:autonomy", correlation_id="corr", trace_id="trace")


def test_node_identity_reconciles_concepts_and_drives_but_not_goals_or_snapshots() -> None:
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)

    rec1 = map_concept_profile_to_substrate(profile=_concept_profile(concept_id="c1", label="Coherence"))
    rec2 = map_concept_profile_to_substrate(profile=_concept_profile(concept_id="c1", label="Coherence"))
    res1 = materializer.apply_record(rec1)
    res2 = materializer.apply_record(rec2)
    assert res1.nodes_created >= 1
    assert res2.nodes_merged >= 1

    now = datetime.now(timezone.utc)
    drive_state = DriveStateV1(
        subject="orion",
        model_layer="autonomy",
        entity_id="entity:orion",
        kind="drive.state",
        provenance=_provenance(),
        pressures={"coherence": 0.8},
        activations={"coherence": True},
        updated_at=now,
    )
    drive_state_2 = DriveStateV1(
        subject="orion",
        model_layer="autonomy",
        entity_id="entity:orion",
        kind="drive.state",
        provenance=_provenance(),
        pressures={"coherence": 0.75},
        activations={"coherence": True},
        updated_at=now + timedelta(seconds=5),
    )
    goal1 = GoalProposalV1(
        subject="orion",
        model_layer="autonomy",
        entity_id="entity:orion",
        kind="goal.proposal",
        provenance=_provenance(),
        goal_statement="stabilize",
        proposal_signature="sig-1",
        drive_origin="coherence",
    )
    goal2 = goal1.model_copy(update={"proposal_signature": "sig-2"})
    auto_rec1 = map_autonomy_artifacts_to_substrate(drive_state=drive_state, goals=[goal1])
    auto_rec2 = map_autonomy_artifacts_to_substrate(drive_state=drive_state_2, goals=[goal2])
    materializer.apply_record(auto_rec1)
    materializer.apply_record(auto_rec2)

    state = store.snapshot()
    goal_nodes = [node for node in state.nodes.values() if node.node_kind == "goal"]
    drive_nodes = [node for node in state.nodes.values() if node.node_kind == "drive"]
    snapshot_nodes = [node for node in state.nodes.values() if node.node_kind == "state_snapshot"]
    assert len(goal_nodes) == 2  # conservative goal identity
    assert len(drive_nodes) >= 1
    assert len(snapshot_nodes) >= 2  # snapshots do not over-collapse


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
    autonomy_record = map_autonomy_artifacts_to_substrate(
        drive_state=DriveStateV1(
            subject="orion",
            model_layer="autonomy",
            entity_id="entity:orion",
            kind="drive.state",
            provenance=_provenance(),
            pressures={"coherence": 0.6},
            activations={"coherence": True},
        )
    )
    out = materializer.apply_records([concept_record, autonomy_record])
    assert len(out) == 2
    assert all(item.nodes_seen >= 1 for item in out)
