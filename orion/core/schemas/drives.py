from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactEventRef(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    event_id: str
    kind: str
    channel: str
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    turn_id: Optional[str] = None
    created_at: Optional[datetime] = None
    source_service: Optional[str] = None

    @field_validator("created_at")
    @classmethod
    def _ensure_created_at_tz(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None or v.tzinfo is not None:
            return v
        return v.replace(tzinfo=timezone.utc)


class ArtifactEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_ref: Optional[ArtifactEventRef] = None
    summary: Optional[str] = None
    text: Optional[str] = None
    source_summary: Optional[str] = None


class ArtifactProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    intake_channel: str
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    turn_id: Optional[str] = None
    evidence_text: Optional[str] = None
    evidence_summary: Optional[str] = None
    source_event_refs: List[ArtifactEventRef] = Field(default_factory=list)
    evidence_items: List[ArtifactEvidence] = Field(default_factory=list)
    tension_refs: List[str] = Field(default_factory=list)


class GraphReadyArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    artifact_id: str = Field(default_factory=lambda: f"artifact-{uuid4()}")
    subject: str
    model_layer: str
    entity_id: str
    kind: str
    ts: datetime = Field(default_factory=_utcnow)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    turn_id: Optional[str] = None
    join_keys: List[str] = Field(default_factory=lambda: ["correlation_id", "trace_id", "turn_id", "artifact_id"])
    provenance: ArtifactProvenance
    related_nodes: List[str] = Field(default_factory=list)

    @field_validator("ts")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class TensionEventV1(GraphReadyArtifact):
    magnitude: float = Field(ge=0.0, le=1.0)
    drive_impacts: Dict[str, float] = Field(default_factory=dict)


class DriveStateV1(GraphReadyArtifact):
    pressures: Dict[str, float] = Field(default_factory=dict)
    activations: Dict[str, bool] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("updated_at")
    @classmethod
    def _ensure_updated_at_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class DriveAuditV1(GraphReadyArtifact):
    drive_pressures: Dict[str, float] = Field(default_factory=dict)
    drive_activations: Dict[str, bool] = Field(default_factory=dict)
    active_drives: List[str] = Field(default_factory=list)
    dominant_drive: Optional[str] = None
    tension_kinds: List[str] = Field(default_factory=list)
    source_event_refs: List[ArtifactEventRef] = Field(default_factory=list)
    evidence_items: List[ArtifactEvidence] = Field(default_factory=list)
    summary: Optional[str] = None


class IdentitySnapshotV1(GraphReadyArtifact):
    anchor_strategy: str
    summary: Optional[str] = None
    source_event_refs: List[ArtifactEventRef] = Field(default_factory=list)
    evidence_items: List[ArtifactEvidence] = Field(default_factory=list)
    tension_kinds: List[str] = Field(default_factory=list)
    drive_pressures: Dict[str, float] = Field(default_factory=dict)


class GoalProposalV1(GraphReadyArtifact):
    goal_statement: str
    proposal_signature: str
    drive_origin: str
    priority: float = Field(default=0.0, ge=0.0, le=1.0)
    cooldown_until: Optional[datetime] = None
    source_event_refs: List[ArtifactEventRef] = Field(default_factory=list)
    evidence_items: List[ArtifactEvidence] = Field(default_factory=list)
    tension_kinds: List[str] = Field(default_factory=list)

    @field_validator("cooldown_until")
    @classmethod
    def _ensure_cooldown_tz(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None or v.tzinfo is not None:
            return v
        return v.replace(tzinfo=timezone.utc)


class TurnDossierV1(BaseModel):
    """Lightweight join stub for turn-level debugability (not a service)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    kind: str = "debug.turn.dossier.v1"
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    turn_id: Optional[str] = None
    subject: Optional[str] = None
    model_layer: Optional[str] = None
    entity_id: Optional[str] = None
    chat_turn_ref: Optional[str] = None
    spark_telemetry_ref: Optional[str] = None
    metacognition_tick_ref: Optional[str] = None
    collapse_sql_write_ref: Optional[str] = None
    cognition_trace_ref: Optional[str] = None
    concept_delta_ref: Optional[str] = None
    source_event_refs: List[ArtifactEventRef] = Field(default_factory=list)
    tension_refs: List[str] = Field(default_factory=list)
    drive_audit_ref: Optional[str] = None
    identity_snapshot_ref: Optional[str] = None
    goal_proposal_ref: Optional[str] = None
    suppressed_goal_signatures: List[str] = Field(default_factory=list)
    join_keys: List[str] = Field(default_factory=lambda: ["correlation_id", "trace_id", "turn_id"])
