from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    intake_channel: str
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    evidence_text: Optional[str] = None


class GraphReadyArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    subject: str
    model_layer: str
    entity_id: str
    kind: str
    ts: datetime = Field(default_factory=_utcnow)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
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


class TurnDossierV1(BaseModel):
    """Lightweight join stub for turn-level debugability (not a service)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    turn_id: Optional[str] = None
    chat_turn_ref: Optional[str] = None
    spark_telemetry_ref: Optional[str] = None
    metacognition_tick_ref: Optional[str] = None
    collapse_sql_write_ref: Optional[str] = None
    cognition_trace_ref: Optional[str] = None
    concept_delta_ref: Optional[str] = None
    join_keys: List[str] = Field(default_factory=lambda: ["correlation_id", "trace_id", "turn_id"])
