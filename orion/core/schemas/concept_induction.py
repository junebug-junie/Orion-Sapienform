"""Canonical schemas for Concept Induction artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_concept_id(label: str) -> str:
    """
    Deterministic-ish concept id helper.

    Uses a simple stable hash of the normalized label to keep ids stable
    across runs while avoiding full-blown UUID collisions for typical use.
    """
    import hashlib

    norm = (label or "").strip().lower()
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"concept-{h[:12]}"


class ConceptEvidenceRef(BaseModel):
    """Lineage reference for evidence used in concept formation."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    message_id: UUID
    correlation_id: Optional[UUID] = None
    timestamp: datetime
    channel: str

    @field_validator("timestamp")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class ConceptItem(BaseModel):
    """A single induced concept or motif."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    concept_id: str = Field(default_factory=lambda: f"concept-{uuid4()}")
    label: str
    aliases: List[str] = Field(default_factory=list)
    type: str = Field(
        default="unknown",
        description="Category such as identity, relationship, motif, or theme.",
    )
    salience: float = Field(default=0.0, ge=0.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding_ref: Optional[str] = Field(
        default=None, description="Reference to embedding vector or store key."
    )
    evidence: List[ConceptEvidenceRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters or notes."
    )


class ConceptCluster(BaseModel):
    """Cluster of related concepts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cluster_id: str = Field(default_factory=lambda: f"cluster-{uuid4()}")
    label: str
    summary: str = ""
    concept_ids: List[str] = Field(default_factory=list)
    cohesion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateEstimate(BaseModel):
    """Trend and trajectory estimates for internal state dynamics."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dimensions: Dict[str, float] = Field(default_factory=dict)
    trend: Dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    window_start: datetime
    window_end: datetime

    @field_validator("window_start", "window_end")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class ConceptProfile(BaseModel):
    """Versioned snapshot of induced concepts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    profile_id: str = Field(default_factory=lambda: f"profile-{uuid4()}")
    subject: str = Field(
        ..., description='One of "orion", "juniper", or "relationship".'
    )
    revision: int = 1
    created_at: datetime = Field(default_factory=_utcnow)
    window_start: datetime
    window_end: datetime
    concepts: List[ConceptItem] = Field(default_factory=list)
    clusters: List[ConceptCluster] = Field(default_factory=list)
    state_estimate: Optional[StateEstimate] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at", "window_start", "window_end")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class ConceptProfileDelta(BaseModel):
    """Delta between two concept profile revisions."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    delta_id: str = Field(default_factory=lambda: f"delta-{uuid4()}")
    profile_id: str
    from_rev: int
    to_rev: int
    added: List[str] = Field(default_factory=list)
    removed: List[str] = Field(default_factory=list)
    updated: List[str] = Field(default_factory=list)
    rationale: str = ""
    evidence: List[ConceptEvidenceRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


ConceptEvidenceRef.model_rebuild()
ConceptItem.model_rebuild()
ConceptCluster.model_rebuild()
StateEstimate.model_rebuild()
ConceptProfile.model_rebuild()
ConceptProfileDelta.model_rebuild()
