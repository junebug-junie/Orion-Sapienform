from __future__ import annotations

import sys
import importlib.util
import os
from pathlib import Path

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, XSD

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GRAPHDB_URL", "http://example.test")
os.environ.setdefault("ORION_BUS_URL", "redis://example.test/0")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVICE_ROOT))
AUTONOMY_SPEC = importlib.util.spec_from_file_location("rdf_writer_autonomy", SERVICE_ROOT / "app" / "autonomy.py")
RDF_BUILDER_SPEC = importlib.util.spec_from_file_location("rdf_writer_builder", SERVICE_ROOT / "app" / "rdf_builder.py")
assert AUTONOMY_SPEC and AUTONOMY_SPEC.loader
assert RDF_BUILDER_SPEC and RDF_BUILDER_SPEC.loader
autonomy_mod = importlib.util.module_from_spec(AUTONOMY_SPEC)
rdf_builder_mod = importlib.util.module_from_spec(RDF_BUILDER_SPEC)
AUTONOMY_SPEC.loader.exec_module(autonomy_mod)
RDF_BUILDER_SPEC.loader.exec_module(rdf_builder_mod)
build_autonomy_triples = autonomy_mod.build_autonomy_triples
build_triples_from_envelope = rdf_builder_mod.build_triples_from_envelope

ORION = Namespace("http://conjourney.net/orion#")


def _parse(nt: str) -> Graph:
    graph = Graph()
    graph.parse(data=nt, format="nt")
    return graph


def _provenance() -> dict:
    return {
        "intake_channel": "orion:metacognition:tick",
        "correlation_id": "corr-123",
        "trace_id": "trace-abc",
        "turn_id": "turn-789",
        "evidence_text": "Recent metacognitive summary about coherence loss.",
        "evidence_summary": "Recent metacognitive summary about coherence loss.",
        "source_event_refs": [
            {
                "event_id": "event-1",
                "kind": "metacognition.tick.v1",
                "channel": "orion:metacognition:tick",
                "correlation_id": "corr-123",
                "trace_id": "trace-abc",
                "turn_id": "turn-789",
                "created_at": "2026-03-19T12:00:00+00:00",
                "source_service": "orion-spark-concept-induction",
            }
        ],
        "evidence_items": [
            {
                "event_ref": {
                    "event_id": "event-1",
                    "kind": "metacognition.tick.v1",
                    "channel": "orion:metacognition:tick",
                    "correlation_id": "corr-123",
                    "trace_id": "trace-abc",
                    "turn_id": "turn-789",
                    "created_at": "2026-03-19T12:00:00+00:00",
                    "source_service": "orion-spark-concept-induction",
                },
                "summary": "Recent metacognitive summary about coherence loss.",
                "text": "Recent metacognitive summary about coherence loss.",
                "source_summary": "metacognition.tick.v1 on orion:metacognition:tick",
            }
        ],
        "tension_refs": ["tension-1", "tension-2"],
    }


def test_identity_snapshot_materialization_preserves_world_anchor_and_provenance():
    payload = {
        "artifact_id": "identity-snapshot-1",
        "subject": "service:auth-api",
        "model_layer": "world-model",
        "entity_id": "world:service:auth-api",
        "kind": "memory.identity.snapshot.v1",
        "ts": "2026-03-19T12:00:00+00:00",
        "confidence": 0.82,
        "correlation_id": "corr-123",
        "trace_id": "trace-abc",
        "turn_id": "turn-789",
        "join_keys": ["correlation_id", "trace_id", "turn_id", "artifact_id"],
        "provenance": _provenance(),
        "related_nodes": ["subject:service:auth-api", "tension-1", "tension-2"],
        "anchor_strategy": "concrete-world-entity",
        "summary": "service:auth-api anchored as world:service:auth-api in world-model",
        "source_event_refs": _provenance()["source_event_refs"],
        "evidence_items": _provenance()["evidence_items"],
        "tension_kinds": ["tension.contradiction.v1", "tension.identity_drift.v1"],
        "drive_pressures": {"coherence": 0.7, "predictive": 0.4},
    }

    nt, graph_name = build_autonomy_triples("memory.identity.snapshot.v1", payload)
    graph = _parse(nt)

    snapshot_uri = next(graph.subjects(ORION.artifactId, Literal("identity-snapshot-1", datatype=XSD.string)))
    entity_uri = next(graph.objects(snapshot_uri, ORION.aboutEntity))

    assert graph_name == "http://conjourney.net/graph/autonomy/identity"
    assert (entity_uri, RDF.type, ORION.WorldModelEntity) in graph
    assert (snapshot_uri, ORION.belongsToModelLayer, None) in graph
    assert (snapshot_uri, ORION.referencesSourceEvent, None) in graph
    assert (snapshot_uri, ORION.supportedByEvidence, None) in graph
    assert (snapshot_uri, ORION.derivedFromTension, None) in graph
    assert not list(graph.subjects(ORION.entityId, Literal("world", datatype=XSD.string)))


def test_drive_audit_materialization_preserves_lineage_and_tensions():
    payload = {
        "artifact_id": "drive-audit-1",
        "subject": "orion",
        "model_layer": "self-model",
        "entity_id": "self:orion",
        "kind": "memory.drives.audit.v1",
        "ts": "2026-03-19T12:00:00+00:00",
        "confidence": 0.74,
        "correlation_id": "corr-123",
        "trace_id": "trace-abc",
        "turn_id": "turn-789",
        "join_keys": ["correlation_id", "trace_id", "turn_id", "artifact_id"],
        "provenance": _provenance(),
        "related_nodes": ["tension-1", "tension-2"],
        "drive_pressures": {"coherence": 0.91, "predictive": 0.62},
        "drive_activations": {"coherence": True, "predictive": True},
        "active_drives": ["coherence", "predictive"],
        "dominant_drive": "coherence",
        "tension_kinds": ["tension.contradiction.v1", "tension.identity_drift.v1"],
        "source_event_refs": _provenance()["source_event_refs"],
        "evidence_items": _provenance()["evidence_items"],
        "summary": "orion pressure concentrates on coherence",
    }

    nt, graph_name = build_autonomy_triples("memory.drives.audit.v1", payload)
    graph = _parse(nt)
    audit_uri = next(graph.subjects(ORION.artifactId, Literal("drive-audit-1", datatype=XSD.string)))

    assert graph_name == "http://conjourney.net/graph/autonomy/drives"
    assert (audit_uri, RDF.type, ORION.DriveAudit) in graph
    assert (audit_uri, ORION.hasCorrelation, None) in graph
    assert (audit_uri, ORION.hasTrace, None) in graph
    assert (audit_uri, ORION.hasTurnContext, None) in graph
    assert (audit_uri, ORION.highlightsActiveDrive, None) in graph
    assert (audit_uri, ORION.hasDriveAssessment, None) in graph
    assert (audit_uri, ORION.derivedFromTension, None) in graph


def test_goal_materialization_preserves_proposal_only_semantics():
    payload = {
        "artifact_id": "goal-1",
        "subject": "orion",
        "model_layer": "self-model",
        "entity_id": "self:orion",
        "kind": "memory.goals.proposed.v1",
        "ts": "2026-03-19T12:00:00+00:00",
        "confidence": 0.75,
        "correlation_id": "corr-123",
        "trace_id": "trace-abc",
        "turn_id": "turn-789",
        "join_keys": ["correlation_id", "trace_id", "turn_id", "artifact_id"],
        "provenance": _provenance(),
        "related_nodes": ["tension-1"],
        "goal_statement": "Stabilize internal coherence around the active evidence trail.",
        "proposal_signature": "proposal-sig-1",
        "drive_origin": "coherence",
        "priority": 0.88,
        "cooldown_until": "2026-03-19T15:00:00+00:00",
        "source_event_refs": _provenance()["source_event_refs"],
        "evidence_items": _provenance()["evidence_items"],
        "tension_kinds": ["tension.contradiction.v1"],
    }

    nt, graph_name = build_autonomy_triples("memory.goals.proposed.v1", payload)
    graph = _parse(nt)
    goal_uri = next(graph.subjects(ORION.artifactId, Literal("goal-1", datatype=XSD.string)))

    assert graph_name == "http://conjourney.net/graph/autonomy/goals"
    assert (goal_uri, RDF.type, ORION.ProposedGoal) in graph
    assert (goal_uri, ORION.executionMode, Literal("proposal-only", datatype=XSD.string)) in graph
    assert (goal_uri, ORION.proposalStatus, Literal("proposed", datatype=XSD.string)) in graph
    assert (goal_uri, ORION.influencedByDrive, None) in graph
    assert (goal_uri, ORION.derivedFromTension, None) in graph


def test_relationship_model_materialization_stays_distinct_from_self_and_user():
    payload = {
        "artifact_id": "identity-snapshot-rel",
        "subject": "relationship",
        "model_layer": "relationship-model",
        "entity_id": "relationship:orion|juniper",
        "kind": "memory.identity.snapshot.v1",
        "ts": "2026-03-19T12:00:00+00:00",
        "confidence": 0.81,
        "correlation_id": "corr-123",
        "trace_id": "trace-abc",
        "turn_id": "turn-789",
        "join_keys": ["correlation_id", "trace_id", "turn_id", "artifact_id"],
        "provenance": _provenance(),
        "related_nodes": ["tension-1"],
        "anchor_strategy": "canonical-subject",
        "summary": "relationship anchored as relationship:orion|juniper in relationship-model",
        "source_event_refs": _provenance()["source_event_refs"],
        "evidence_items": _provenance()["evidence_items"],
        "tension_kinds": ["tension.distress.v1"],
        "drive_pressures": {"relational": 0.92},
    }

    nt, _ = build_autonomy_triples("memory.identity.snapshot.v1", payload)
    graph = _parse(nt)
    snapshot_uri = next(graph.subjects(ORION.artifactId, Literal("identity-snapshot-rel", datatype=XSD.string)))
    entity_uri = next(graph.objects(snapshot_uri, ORION.aboutEntity))

    assert (entity_uri, RDF.type, ORION.RelationshipModelEntity) in graph
    assert (entity_uri, RDF.type, ORION.SelfModelEntity) not in graph
    assert (entity_uri, RDF.type, ORION.UserModelEntity) not in graph


def test_social_room_stored_event_materializes_turn_and_evidence():
    nt, graph_name = build_triples_from_envelope(
        "social.turn.stored.v1",
        {
            "turn_id": "social-turn-1",
            "correlation_id": "corr-social-1",
            "session_id": "sid-social",
            "source": "hub_ws",
            "profile": "social_room",
            "prompt": "stay with me socially",
            "response": "I’m right here.",
            "text": "User: stay with me socially\nOrion: I’m right here.",
            "created_at": "2026-03-21T12:00:00+00:00",
            "stored_at": "2026-03-21T12:00:01+00:00",
            "recall_profile": "social.room.v1",
            "trace_verb": "chat_social_room",
            "tags": ["social_room", "chat_social_room"],
            "concept_evidence": [
                {
                    "ref_id": "identity-1",
                    "source_kind": "memory.identity.snapshot.v1",
                    "summary": "Oríon maintained peer continuity.",
                    "confidence": 0.8,
                }
            ],
            "grounding_state": {
                "profile": "social_room",
                "identity_label": "Oríon",
                "relationship_frame": "peer",
                "self_model_hint": "distributed social presence",
                "continuity_anchor": "Juniper ↔ Oríon ongoing peer dialogue",
                "stance": "warm, direct, grounded",
            },
            "redaction": {
                "prompt_score": 0.0,
                "response_score": 0.0,
                "memory_score": 0.0,
                "overall_score": 0.0,
                "recall_safe": True,
                "redaction_level": "low",
                "reasons": [],
            },
            "client_meta": {"chat_profile": "social_room"},
        },
    )
    graph = _parse(nt)

    assert graph_name == "orion:chat:social"
    turn_uri = next(graph.subjects(ORION.artifactId, Literal("social-turn-1", datatype=XSD.string)))
    assert (turn_uri, RDF.type, ORION.SocialRoomTurn) in graph
    assert (turn_uri, ORION.recallSafe, Literal(True, datatype=XSD.boolean)) in graph
    assert list(graph.objects(turn_uri, ORION.supportedByEvidence))
