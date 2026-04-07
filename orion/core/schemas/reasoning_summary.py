"""Compiled reasoning-summary contracts for turn-time stance consumption (Phase 4)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReasoningSummaryRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(default_factory=lambda: f"reasoning-summary-{uuid4()}")
    anchor_scope: Literal["orion", "juniper", "relationship", "world", "session"] = "orion"
    subject_refs: list[str] = Field(default_factory=list)
    include_provisional: bool = True
    max_claims: int = Field(default=6, ge=1, le=20)
    compiled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("compiled_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class ReasoningClaimDigestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    artifact_id: str
    anchor_scope: Literal["orion", "juniper", "relationship", "world", "session"]
    subject_ref: Optional[str] = None
    status: Literal["proposed", "provisional", "canonical", "deprecated", "rejected"]
    claim_kind: str
    claim_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReasoningConceptDigestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    artifact_id: str
    concept_id: str
    label: str
    concept_type: str
    anchor_scope: Literal["orion", "juniper", "relationship", "world", "session"]
    subject_ref: Optional[str] = None
    status: Literal["proposed", "provisional", "canonical", "deprecated", "rejected"]
    confidence: float = Field(ge=0.0, le=1.0)
    salience: float = Field(ge=0.0, le=1.0)


class ReasoningSparkSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    present: bool = False
    observed_at: Optional[datetime] = None
    dimensions: dict[str, float] = Field(default_factory=dict)
    tensions: list[str] = Field(default_factory=list)


class ReasoningAutonomySummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    present: bool = False
    posture: list[str] = Field(default_factory=list)
    active_goals: list[str] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list)


class ReasoningSummaryDebugV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    compiler_ran: bool = False
    compiler_succeeded: bool = False
    fallback_used: bool = True
    considered_count: int = 0
    included_count: int = 0
    suppressed_count: int = 0
    suppressed_by_contradiction: int = 0
    suppressed_by_drift: int = 0
    suppressed_by_status: int = 0
    selected_anchor_scopes: list[str] = Field(default_factory=list)
    selected_subject_refs: list[str] = Field(default_factory=list)


class ReasoningSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str
    anchor_scope: Literal["orion", "juniper", "relationship", "world", "session"]
    active_subject_refs: list[str] = Field(default_factory=list)
    active_claims: list[ReasoningClaimDigestV1] = Field(default_factory=list)
    active_concepts: list[ReasoningConceptDigestV1] = Field(default_factory=list)
    relationship_signals: list[str] = Field(default_factory=list)
    tensions: list[str] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list)
    autonomy: ReasoningAutonomySummaryV1 = Field(default_factory=ReasoningAutonomySummaryV1)
    spark: ReasoningSparkSummaryV1 = Field(default_factory=ReasoningSparkSummaryV1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    fallback_recommended: bool = True
    debug: ReasoningSummaryDebugV1 = Field(default_factory=ReasoningSummaryDebugV1)
