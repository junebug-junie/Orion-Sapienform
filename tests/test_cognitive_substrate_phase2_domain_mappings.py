from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from orion.core.schemas.concept_induction import ConceptCluster, ConceptEvidenceRef, ConceptItem, ConceptProfile, ConceptProfileDelta
from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
from orion.core.schemas.drives import ArtifactProvenance, DriveAuditV1, DriveStateV1, GoalProposalV1, IdentitySnapshotV1, TensionEventV1
from orion.core.schemas.spark_canonical import SparkSourceSnapshotV1
from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.substrate.adapters import (
    map_autonomy_artifacts_to_substrate,
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


def _provenance() -> ArtifactProvenance:
    return ArtifactProvenance(intake_channel="orion:autonomy", correlation_id="corr-1", trace_id="trace-1")


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


def test_concept_delta_adapter_only_emits_contradiction_when_semantics_support_it() -> None:
    now = datetime.now(timezone.utc)
    delta = ConceptProfileDelta(profile_id="p1", from_rev=1, to_rev=2, added=["c2"], removed=["c1"], rationale="concept conflict")
    out = map_concept_delta_to_substrate(delta=delta, observed_at=now, anchor_scope="orion", subject_ref="project:orion")
    contradiction_nodes = [n for n in out.nodes if n.node_kind == "contradiction"]
    assert len(contradiction_nodes) == 1
    assert all(edge.predicate == "contradicts" for edge in out.edges)


def test_autonomy_adapter_maps_drive_goal_tension_state_without_overpromoting() -> None:
    now = datetime.now(timezone.utc)
    drive_audit = DriveAuditV1(
        subject="orion",
        model_layer="autonomy",
        entity_id="entity:orion",
        kind="drive.audit",
        provenance=_provenance(),
        drive_pressures={"coherence": 0.8},
        active_drives=["coherence"],
        tension_kinds=["goal_conflict"],
    )
    drive_state = DriveStateV1(
        subject="orion",
        model_layer="autonomy",
        entity_id="entity:orion",
        kind="drive.state",
        provenance=_provenance(),
        pressures={"coherence": 0.9},
        activations={"coherence": True},
    )
    goal = GoalProposalV1(
        subject="orion",
        model_layer="autonomy",
        entity_id="entity:orion",
        kind="goal.proposal",
        provenance=_provenance(),
        goal_statement="stabilize context",
        proposal_signature="sig-1",
        drive_origin="coherence",
        priority=0.7,
        tension_kinds=["goal_conflict"],
    )
    tension = TensionEventV1(
        subject="orion",
        model_layer="autonomy",
        entity_id="entity:orion",
        kind="goal_conflict",
        provenance=_provenance(),
        magnitude=0.6,
        drive_impacts={"coherence": 0.5},
    )
    identity = IdentitySnapshotV1(
        subject="orion",
        model_layer="autonomy",
        entity_id="entity:orion",
        kind="identity.snapshot",
        provenance=_provenance(),
        anchor_strategy="steady",
        drive_pressures={"coherence": 0.7},
    )
    out = map_autonomy_artifacts_to_substrate(
        drive_audit=drive_audit,
        drive_state=drive_state,
        goals=[goal],
        tensions=[tension],
        identity_snapshot=identity,
        anchor_scope="orion",
    )
    assert isinstance(out, SubstrateGraphRecordV1)
    assert any(n.node_kind == "drive" for n in out.nodes)
    assert any(n.node_kind == "goal" for n in out.nodes)
    assert any(n.node_kind == "tension" for n in out.nodes)
    assert any(n.node_kind == "state_snapshot" for n in out.nodes)
    assert any(e.predicate == "seeks" for e in out.edges)
    assert all(n.node_kind != "ontology_branch" for n in out.nodes)


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
