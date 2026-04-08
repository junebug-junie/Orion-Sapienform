from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from orion.core.schemas.drives import (
    ArtifactEvidence,
    ArtifactEventRef,
    DriveAuditV1,
    GoalProposalV1,
    GraphReadyArtifact,
    IdentitySnapshotV1,
)

ORION = Namespace("http://conjourney.net/orion#")
PROV = Namespace("http://www.w3.org/ns/prov#")
BASE = "http://conjourney.net/orion/autonomy"


def _sanitize_fragment(raw: object) -> str:
    return "".join(c if str(c).isalnum() else "_" for c in str(raw or "unknown"))


def _literal_if(value: object, datatype=None) -> Optional[Literal]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return Literal(value, datatype=datatype)


def _graph_uri(name: str) -> str:
    return f"http://conjourney.net/graph/autonomy/{name}"


def _artifact_uri(kind: str, artifact_id: str) -> URIRef:
    suffix = {
        "memory.identity.snapshot.v1": "identitySnapshot",
        "memory.drives.audit.v1": "driveAudit",
        "memory.goals.proposed.v1": "proposedGoal",
    }.get(kind, "artifact")
    return URIRef(f"{BASE}/{suffix}/{_sanitize_fragment(artifact_id)}")


def _entity_uri(entity_id: str) -> URIRef:
    return URIRef(f"http://conjourney.net/orion/entity/{_sanitize_fragment(entity_id)}")


def _layer_uri(model_layer: str) -> URIRef:
    return URIRef(f"http://conjourney.net/orion/modelLayer/{_sanitize_fragment(model_layer)}")


def _drive_dimension_uri(drive_name: str) -> URIRef:
    return URIRef(f"{BASE}/driveDimension/{_sanitize_fragment(drive_name)}")


def _lineage_uri(kind: str, value: str) -> URIRef:
    return URIRef(f"{BASE}/{kind}/{_sanitize_fragment(value)}")


def _event_ref_uri(event_id: str) -> URIRef:
    return URIRef(f"{BASE}/sourceEvent/{_sanitize_fragment(event_id)}")


def _evidence_uri(artifact_id: str, index: int) -> URIRef:
    return URIRef(f"{BASE}/evidence/{_sanitize_fragment(artifact_id)}/{index}")


def _tension_uri(tension_ref: str) -> URIRef:
    return URIRef(f"{BASE}/tension/{_sanitize_fragment(tension_ref)}")


def _provenance_uri(artifact_id: str) -> URIRef:
    return URIRef(f"{BASE}/provenance/{_sanitize_fragment(artifact_id)}")


def _entity_type(model_layer: str) -> URIRef:
    return {
        "self-model": ORION.SelfModelEntity,
        "user-model": ORION.UserModelEntity,
        "world-model": ORION.WorldModelEntity,
        "relationship-model": ORION.RelationshipModelEntity,
    }.get(model_layer, ORION.ModelEntity)


def _artifact_type(kind: str) -> URIRef:
    return {
        "memory.identity.snapshot.v1": ORION.IdentitySnapshot,
        "memory.drives.audit.v1": ORION.DriveAudit,
        "memory.goals.proposed.v1": ORION.ProposedGoal,
    }[kind]


def _add_literal(g: Graph, subject: URIRef, predicate: URIRef, value: object, datatype=None) -> None:
    lit = _literal_if(value, datatype=datatype)
    if lit is not None:
        g.add((subject, predicate, lit))


def _add_lineage(g: Graph, artifact_uri: URIRef, artifact: GraphReadyArtifact) -> None:
    if artifact.correlation_id:
        corr_uri = _lineage_uri("correlation", artifact.correlation_id)
        g.add((corr_uri, RDF.type, ORION.CorrelationThread))
        _add_literal(g, corr_uri, ORION.correlationId, artifact.correlation_id, XSD.string)
        g.add((artifact_uri, ORION.hasCorrelation, corr_uri))
    if artifact.trace_id:
        trace_uri = _lineage_uri("trace", artifact.trace_id)
        g.add((trace_uri, RDF.type, ORION.TraceSpan))
        _add_literal(g, trace_uri, ORION.traceId, artifact.trace_id, XSD.string)
        g.add((artifact_uri, ORION.hasTrace, trace_uri))
    if artifact.turn_id:
        turn_uri = _lineage_uri("turn", artifact.turn_id)
        g.add((turn_uri, RDF.type, ORION.TurnContext))
        _add_literal(g, turn_uri, ORION.turnId, artifact.turn_id, XSD.string)
        g.add((artifact_uri, ORION.hasTurnContext, turn_uri))


def _add_source_event_ref(g: Graph, artifact_uri: URIRef, event_ref: ArtifactEventRef) -> URIRef:
    event_uri = _event_ref_uri(event_ref.event_id)
    g.add((event_uri, RDF.type, ORION.SourceEventRef))
    _add_literal(g, event_uri, ORION.eventId, event_ref.event_id, XSD.string)
    _add_literal(g, event_uri, ORION.eventKind, event_ref.kind, XSD.string)
    _add_literal(g, event_uri, ORION.busChannel, event_ref.channel, XSD.string)
    _add_literal(g, event_uri, ORION.sourceService, event_ref.source_service, XSD.string)
    _add_literal(g, event_uri, ORION.createdAt, event_ref.created_at.isoformat() if event_ref.created_at else None, XSD.dateTime)
    if event_ref.correlation_id:
        corr_uri = _lineage_uri("correlation", event_ref.correlation_id)
        g.add((corr_uri, RDF.type, ORION.CorrelationThread))
        _add_literal(g, corr_uri, ORION.correlationId, event_ref.correlation_id, XSD.string)
        g.add((event_uri, ORION.hasCorrelation, corr_uri))
    if event_ref.trace_id:
        trace_uri = _lineage_uri("trace", event_ref.trace_id)
        g.add((trace_uri, RDF.type, ORION.TraceSpan))
        _add_literal(g, trace_uri, ORION.traceId, event_ref.trace_id, XSD.string)
        g.add((event_uri, ORION.hasTrace, trace_uri))
    if event_ref.turn_id:
        turn_uri = _lineage_uri("turn", event_ref.turn_id)
        g.add((turn_uri, RDF.type, ORION.TurnContext))
        _add_literal(g, turn_uri, ORION.turnId, event_ref.turn_id, XSD.string)
        g.add((event_uri, ORION.hasTurnContext, turn_uri))
    g.add((artifact_uri, ORION.referencesSourceEvent, event_uri))
    g.add((artifact_uri, PROV.wasDerivedFrom, event_uri))
    return event_uri


def _add_evidence(g: Graph, artifact_uri: URIRef, artifact_id: str, evidence_items: Sequence[ArtifactEvidence]) -> None:
    for index, evidence in enumerate(evidence_items):
        evidence_uri = _evidence_uri(artifact_id, index)
        g.add((evidence_uri, RDF.type, ORION.EvidenceItem))
        _add_literal(g, evidence_uri, ORION.evidenceSummary, evidence.summary, XSD.string)
        _add_literal(g, evidence_uri, ORION.evidenceText, evidence.text, XSD.string)
        _add_literal(g, evidence_uri, ORION.sourceSummary, evidence.source_summary, XSD.string)
        g.add((artifact_uri, ORION.supportedByEvidence, evidence_uri))
        if evidence.event_ref:
            event_uri = _add_source_event_ref(g, artifact_uri, evidence.event_ref)
            g.add((evidence_uri, ORION.referencesSourceEvent, event_uri))
            g.add((evidence_uri, PROV.wasDerivedFrom, event_uri))


def _add_tensions(g: Graph, artifact_uri: URIRef, tension_refs: Iterable[str], tension_kinds: Sequence[str]) -> None:
    tension_list = list(tension_refs)
    for index, tension_ref in enumerate(tension_list):
        tension_uri = _tension_uri(tension_ref)
        g.add((tension_uri, RDF.type, ORION.TensionReference))
        _add_literal(g, tension_uri, ORION.tensionRefId, tension_ref, XSD.string)
        kind = tension_kinds[index] if index < len(tension_kinds) else None
        _add_literal(g, tension_uri, ORION.tensionKind, kind, XSD.string)
        g.add((artifact_uri, ORION.derivedFromTension, tension_uri))
        g.add((artifact_uri, PROV.wasDerivedFrom, tension_uri))


def _add_common_artifact(g: Graph, artifact: GraphReadyArtifact) -> Tuple[URIRef, URIRef, URIRef]:
    artifact_uri = _artifact_uri(artifact.kind, artifact.artifact_id)
    entity_uri = _entity_uri(artifact.entity_id)
    layer_uri = _layer_uri(artifact.model_layer)

    g.add((artifact_uri, RDF.type, ORION.AutonomyArtifact))
    g.add((artifact_uri, RDF.type, _artifact_type(artifact.kind)))
    _add_literal(g, artifact_uri, ORION.artifactId, artifact.artifact_id, XSD.string)
    _add_literal(g, artifact_uri, ORION.artifactKind, artifact.kind, XSD.string)
    _add_literal(g, artifact_uri, ORION.subjectKey, artifact.subject, XSD.string)
    _add_literal(g, artifact_uri, ORION.entityId, artifact.entity_id, XSD.string)
    _add_literal(g, artifact_uri, ORION.modelLayerKey, artifact.model_layer, XSD.string)
    _add_literal(g, artifact_uri, ORION.confidence, artifact.confidence, XSD.float)
    _add_literal(g, artifact_uri, ORION.timestamp, artifact.ts.isoformat(), XSD.dateTime)

    g.add((layer_uri, RDF.type, ORION.ModelLayer))
    _add_literal(g, layer_uri, RDFS.label, artifact.model_layer, XSD.string)

    g.add((entity_uri, RDF.type, ORION.ModelEntity))
    g.add((entity_uri, RDF.type, _entity_type(artifact.model_layer)))
    _add_literal(g, entity_uri, RDFS.label, artifact.entity_id, XSD.string)
    _add_literal(g, entity_uri, ORION.entityId, artifact.entity_id, XSD.string)
    _add_literal(g, entity_uri, ORION.subjectKey, artifact.subject, XSD.string)
    _add_literal(g, entity_uri, ORION.modelLayerKey, artifact.model_layer, XSD.string)
    g.add((entity_uri, ORION.inModelLayer, layer_uri))

    g.add((artifact_uri, ORION.aboutEntity, entity_uri))
    g.add((artifact_uri, ORION.belongsToModelLayer, layer_uri))
    _add_lineage(g, artifact_uri, artifact)

    provenance_uri = _provenance_uri(artifact.artifact_id)
    g.add((provenance_uri, RDF.type, ORION.ArtifactProvenance))
    _add_literal(g, provenance_uri, ORION.intakeChannel, artifact.provenance.intake_channel, XSD.string)
    _add_literal(g, provenance_uri, ORION.evidenceSummary, artifact.provenance.evidence_summary, XSD.string)
    _add_literal(g, provenance_uri, ORION.evidenceText, artifact.provenance.evidence_text, XSD.string)
    g.add((artifact_uri, ORION.hasProvenance, provenance_uri))

    for event_ref in artifact.provenance.source_event_refs:
        event_uri = _add_source_event_ref(g, artifact_uri, event_ref)
        g.add((provenance_uri, ORION.referencesSourceEvent, event_uri))
    _add_evidence(g, artifact_uri, artifact.artifact_id, artifact.provenance.evidence_items)

    return artifact_uri, entity_uri, provenance_uri


def _add_drive_dimension(g: Graph, drive_name: str) -> URIRef:
    drive_uri = _drive_dimension_uri(drive_name)
    g.add((drive_uri, RDF.type, ORION.DriveDimension))
    _add_literal(g, drive_uri, RDFS.label, drive_name, XSD.string)
    return drive_uri


def _handle_identity_snapshot(g: Graph, snapshot: IdentitySnapshotV1) -> Tuple[str, str]:
    artifact_uri, _, provenance_uri = _add_common_artifact(g, snapshot)
    _add_literal(g, artifact_uri, ORION.anchorStrategy, snapshot.anchor_strategy, XSD.string)
    _add_literal(g, artifact_uri, ORION.snapshotSummary, snapshot.summary, XSD.string)
    _add_tensions(g, artifact_uri, snapshot.provenance.tension_refs, snapshot.tension_kinds)
    for drive_name, pressure in sorted(snapshot.drive_pressures.items()):
        drive_uri = _add_drive_dimension(g, drive_name)
        g.add((artifact_uri, ORION.referencesDriveDimension, drive_uri))
        assessment_uri = URIRef(f"{artifact_uri}/drive/{_sanitize_fragment(drive_name)}")
        g.add((assessment_uri, RDF.type, ORION.DriveAssessment))
        g.add((assessment_uri, ORION.driveDimension, drive_uri))
        _add_literal(g, assessment_uri, ORION.drivePressure, pressure, XSD.float)
        g.add((artifact_uri, ORION.hasDriveAssessment, assessment_uri))
        g.add((provenance_uri, ORION.referencesDriveDimension, drive_uri))
    return g.serialize(format="nt"), _graph_uri("identity")


def _handle_drive_audit(g: Graph, audit: DriveAuditV1) -> Tuple[str, str]:
    artifact_uri, _, provenance_uri = _add_common_artifact(g, audit)
    _add_literal(g, artifact_uri, ORION.auditSummary, audit.summary, XSD.string)
    _add_literal(g, artifact_uri, ORION.dominantDriveName, audit.dominant_drive, XSD.string)
    _add_tensions(g, artifact_uri, audit.provenance.tension_refs, audit.tension_kinds)
    for drive_name, pressure in sorted(audit.drive_pressures.items()):
        drive_uri = _add_drive_dimension(g, drive_name)
        g.add((artifact_uri, ORION.referencesDriveDimension, drive_uri))
        assessment_uri = URIRef(f"{artifact_uri}/drive/{_sanitize_fragment(drive_name)}")
        g.add((assessment_uri, RDF.type, ORION.DriveAssessment))
        g.add((assessment_uri, ORION.driveDimension, drive_uri))
        _add_literal(g, assessment_uri, ORION.drivePressure, pressure, XSD.float)
        _add_literal(g, assessment_uri, ORION.driveActive, bool(audit.drive_activations.get(drive_name)), XSD.boolean)
        g.add((artifact_uri, ORION.hasDriveAssessment, assessment_uri))
        g.add((provenance_uri, ORION.referencesDriveDimension, drive_uri))
    for active_drive in audit.active_drives:
        drive_uri = _add_drive_dimension(g, active_drive)
        g.add((artifact_uri, ORION.highlightsActiveDrive, drive_uri))
    return g.serialize(format="nt"), _graph_uri("drives")


def _handle_goal_proposal(g: Graph, goal: GoalProposalV1) -> Tuple[str, str]:
    artifact_uri, _, _ = _add_common_artifact(g, goal)
    _add_literal(g, artifact_uri, ORION.goalStatement, goal.goal_statement, XSD.string)
    _add_literal(g, artifact_uri, ORION.proposalSignature, goal.proposal_signature, XSD.string)
    _add_literal(g, artifact_uri, ORION.driveOrigin, goal.drive_origin, XSD.string)
    _add_literal(g, artifact_uri, ORION.proposalPriority, goal.priority, XSD.float)
    _add_literal(g, artifact_uri, ORION.cooldownUntil, goal.cooldown_until.isoformat() if goal.cooldown_until else None, XSD.dateTime)
    _add_literal(g, artifact_uri, ORION.proposalStatus, "proposed", XSD.string)
    _add_literal(g, artifact_uri, ORION.executionMode, "proposal-only", XSD.string)
    drive_uri = _add_drive_dimension(g, goal.drive_origin)
    g.add((artifact_uri, ORION.influencedByDrive, drive_uri))
    _add_tensions(g, artifact_uri, goal.provenance.tension_refs, goal.tension_kinds)
    return g.serialize(format="nt"), _graph_uri("goals")


def build_autonomy_triples(env_kind: str, payload: object) -> Tuple[Optional[str], Optional[str]]:
    g = Graph()
    g.bind("orion", ORION)
    g.bind("prov", PROV)

    if env_kind == "memory.identity.snapshot.v1":
        snapshot = IdentitySnapshotV1.model_validate(payload)
        return _handle_identity_snapshot(g, snapshot)
    if env_kind == "memory.drives.audit.v1":
        audit = DriveAuditV1.model_validate(payload)
        return _handle_drive_audit(g, audit)
    if env_kind == "memory.goals.proposed.v1":
        goal = GoalProposalV1.model_validate(payload)
        return _handle_goal_proposal(g, goal)
    return None, None
