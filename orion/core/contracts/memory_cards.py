from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MEMORY_CARD_TYPE_TOKENS = frozenset(
    {
        "anchor",
        "fact",
        "preference",
        "episodic",
        "meta",
        "belief",
        "health",
        "relationship",
        "project",
        "place",
        "event",
        "concept",
    }
)

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
        if self.kind == "era_bound" and not (self.start and str(self.start).strip()):
            raise ValueError("era_bound requires start")
        return self


class EvidenceItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    excerpt: Optional[str] = None
    ts: Optional[str] = None


class MemoryCardV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    card_id: Optional[UUID] = None
    slug: str
    types: List[str]
    anchor_class: Optional[str] = None
    status: str = "pending_review"
    confidence: str = "likely"
    sensitivity: str = "private"
    priority: str = "episodic_detail"
    visibility_scope: List[str] = Field(default_factory=lambda: ["chat"])
    time_horizon: Optional[TimeHorizonV1] = None
    provenance: str
    trust_source: Optional[str] = None
    project: Optional[str] = None
    title: str
    summary: str
    still_true: Optional[List[str]] = None
    anchors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    evidence: List[EvidenceItemV1] = Field(default_factory=list)
    subschema: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("types")
    @classmethod
    def _types_tokens(cls, v: List[str]) -> List[str]:
        for t in v:
            if t not in MEMORY_CARD_TYPE_TOKENS:
                raise ValueError(f"unknown memory card type token: {t}")
        return v

    @model_validator(mode="after")
    def _anchor_class_rule(self) -> MemoryCardV1:
        if "anchor" in self.types and not (self.anchor_class and str(self.anchor_class).strip()):
            raise ValueError("anchor_class required when 'anchor' in types")
        return self


class MemoryCardEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    edge_id: Optional[UUID] = None
    from_card_id: UUID
    to_card_id: UUID
    edge_type: EDGE_TYPES
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class MemoryCardHistoryEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    history_id: UUID
    card_id: Optional[UUID] = None
    edge_id: Optional[UUID] = None
    op: str
    actor: str
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    created_at: datetime


def visibility_allows_card(visibility_scope: List[str], lane: Optional[str]) -> bool:
    scopes = [str(s) for s in (visibility_scope or [])]
    if "all" in scopes:
        return True
    if lane is None:
        return True
    return lane in scopes
