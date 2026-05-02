from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MemoryCardStatus = Literal["pending_review", "active", "rejected", "superseded", "archived", "deprecated"]
MemoryConfidence = Literal["certain", "likely", "possible", "uncertain"]
MemorySensitivity = Literal["public", "private", "intimate"]
MemoryPriority = Literal["always_inject", "high_recall", "episodic_detail", "archival"]
MemoryProvenance = Literal["operator_highlight", "operator_distiller", "auto_extractor", "imported"]
AnchorClass = Literal[
    "person",
    "place",
    "project",
    "event",
    "concept",
    "relationship",
    "health",
    "preference",
    "belief",
]

EDGE_TYPES = Literal[
    "relates_to",
    "contradicts",
    "supersedes",
    "supports",
    "parent_of",
    "child_of",
    "precedes",
    "follows",
    "co_occurs_with",
    "derived_from",
    "evidence_for",
    "evidence_against",
    "tagged_as",
    "instance_of",
    "example_of",
    "analogy_of",
    "associated_with",
]

HISTORY_OPS = Literal[
    "create",
    "update",
    "status_change",
    "edge_add",
    "edge_remove",
    "reverse_auto_promotion",
]


class TimeHorizonV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["timeless", "era_bound", "current", "expiring"]
    start: Optional[str] = None
    end: Optional[str] = None
    as_of: Optional[str] = None

    @model_validator(mode="after")
    def _era_requires_start(self) -> TimeHorizonV1:
        if self.kind == "era_bound" and not (self.start or "").strip():
            raise ValueError("era_bound requires start")
        return self


class EvidenceItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    excerpt: Optional[str] = None
    ts: Optional[str] = None


class MemoryCardV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    card_id: UUID
    slug: str
    types: List[str]
    anchor_class: Optional[str] = None
    status: MemoryCardStatus = "pending_review"
    confidence: MemoryConfidence = "likely"
    sensitivity: MemorySensitivity = "private"
    priority: MemoryPriority = "episodic_detail"
    visibility_scope: List[str] = Field(default_factory=lambda: ["chat"])
    time_horizon: Optional[TimeHorizonV1] = None
    provenance: MemoryProvenance
    trust_source: Optional[str] = None
    project: Optional[str] = None
    title: str
    summary: str
    still_true: Optional[List[str]] = None
    anchors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    evidence: List[EvidenceItemV1] = Field(default_factory=list)
    subschema: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @field_validator("types")
    @classmethod
    def _types_nonempty(cls, v: List[str]) -> List[str]:
        out = [str(t).strip() for t in v if str(t).strip()]
        if not out:
            raise ValueError("types must be non-empty")
        return out

    @model_validator(mode="after")
    def _anchor_class_when_anchor_type(self) -> MemoryCardV1:
        if "anchor" in self.types and not (self.anchor_class or "").strip():
            raise ValueError("anchor_class required when 'anchor' in types")
        return self


class MemoryCardCreateV1(BaseModel):
    """Payload for creating a card (Hub POST)."""

    model_config = ConfigDict(extra="forbid")

    slug: Optional[str] = None
    types: List[str]
    anchor_class: Optional[str] = None
    status: MemoryCardStatus = "pending_review"
    confidence: MemoryConfidence = "likely"
    sensitivity: MemorySensitivity = "private"
    priority: MemoryPriority = "episodic_detail"
    visibility_scope: List[str] = Field(default_factory=lambda: ["chat"])
    time_horizon: Optional[TimeHorizonV1] = None
    provenance: MemoryProvenance = "operator_highlight"
    trust_source: Optional[str] = None
    project: Optional[str] = None
    title: str
    summary: str
    still_true: Optional[List[str]] = None
    anchors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    evidence: List[EvidenceItemV1] = Field(default_factory=list)
    subschema: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("types")
    @classmethod
    def _types_nonempty(cls, v: List[str]) -> List[str]:
        out = [str(t).strip() for t in v if str(t).strip()]
        if not out:
            raise ValueError("types must be non-empty")
        return out

    @model_validator(mode="after")
    def _anchor_class_when_anchor_type(self) -> MemoryCardCreateV1:
        if "anchor" in self.types and not (self.anchor_class or "").strip():
            raise ValueError("anchor_class required when 'anchor' in types")
        return self


class MemoryCardPatchV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: Optional[str] = None
    types: Optional[List[str]] = None
    anchor_class: Optional[str] = None
    status: Optional[MemoryCardStatus] = None
    confidence: Optional[MemoryConfidence] = None
    sensitivity: Optional[MemorySensitivity] = None
    priority: Optional[MemoryPriority] = None
    visibility_scope: Optional[List[str]] = None
    time_horizon: Optional[TimeHorizonV1] = None
    provenance: Optional[MemoryProvenance] = None
    trust_source: Optional[str] = None
    project: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    still_true: Optional[List[str]] = None
    anchors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    evidence: Optional[List[EvidenceItemV1]] = None
    subschema: Optional[Dict[str, Any]] = None

    @field_validator("types")
    @classmethod
    def _types_nonempty(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        out = [str(t).strip() for t in v if str(t).strip()]
        if not out:
            raise ValueError("types must be non-empty")
        return out


class MemoryCardEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    edge_id: UUID
    from_card_id: UUID
    to_card_id: UUID
    edge_type: EDGE_TYPES
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class MemoryCardHistoryEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    history_id: UUID
    card_id: Optional[UUID] = None
    edge_id: Optional[UUID] = None
    op: HISTORY_OPS
    actor: str
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    created_at: datetime


class MemoryCardEdgeCreateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_card_id: UUID
    to_card_id: UUID
    edge_type: EDGE_TYPES
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryCardStatusChangeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: MemoryCardStatus
    reason: Optional[str] = None


def visibility_allows_card(*, lane: Optional[str], visibility_scope: List[str]) -> bool:
    if "all" in visibility_scope:
        return True
    if lane is None:
        return True
    return lane in visibility_scope
