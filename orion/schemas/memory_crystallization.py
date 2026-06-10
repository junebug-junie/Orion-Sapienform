"""MemoryCrystallizationV1 — governed cognitive memory artifacts.

MemoryCardV1 (orion/core/contracts/memory_cards.py) remains the turn-facing
recall artifact. GrammarEventV1 (orion/schemas/grammar.py) remains the
substrate trace artifact. MemoryCrystallizationV1 is the governed,
evidence-backed, contradiction-aware crystallization layer above both.

These schemas do not redefine grammar law: the grammar envelope only
references existing grammar artifact ids (GrammarEventV1 / GrammarAtomV1 /
GrammarEdgeV1).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

CrystallizationKind = Literal[
    "semantic",
    "episode",
    "procedure",
    "stance",
    "open_loop",
    "contradiction",
    "attractor",
    "decision",
    "failure_mode",
]

CrystallizationStatus = Literal[
    "proposed",
    "active",
    "rejected",
    "superseded",
    "deprecated",
    "archived",
    "quarantined",
]

CrystallizationConfidence = Literal[
    "certain",
    "likely",
    "possible",
    "uncertain",
]

CrystallizationSourceKind = Literal[
    "memory_card",
    "grammar_event",
    "grammar_atom",
    "grammar_edge",
    "chat_turn",
    "tool_result",
    "service_trace",
    "repo_event",
    "rdf_memory_graph",
    "graphiti_episode",
    "operator_note",
]

CrystallizationRelation = Literal[
    "supports",
    "contradicts",
    "supersedes",
    "narrows",
    "expands",
    "refines",
    "depends_on",
    "derived_from",
    "evidence_for",
    "evidence_against",
    "co_occurs_with",
    "related_to",
]


def new_crystallization_id() -> str:
    return f"crys_{uuid4().hex}"


class CrystallizationEvidenceRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_kind: CrystallizationSourceKind
    source_id: str
    excerpt: str | None = None
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    note: str | None = None


class CrystallizationClaimV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim: str
    status: Literal["active", "tentative", "rejected", "superseded"] = "active"
    confidence: CrystallizationConfidence = "likely"
    evidence_refs: list[str] = Field(default_factory=list)


class MemoryGrammarEnvelopeV1(BaseModel):
    """Memory-local adornment referencing existing grammar artifacts.

    Not a grammar event schema; grammar law stays in orion/schemas/grammar.py.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["memory_grammar_envelope.v1"] = "memory_grammar_envelope.v1"

    source_grammar_event_ids: list[str] = Field(default_factory=list)
    source_atom_ids: list[str] = Field(default_factory=list)
    source_edge_ids: list[str] = Field(default_factory=list)

    shape: str | None = None
    atom_keys: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)

    planning_effects: list[str] = Field(default_factory=list)
    retrieval_affordances: list[str] = Field(default_factory=list)

    field_effects: list[dict[str, Any]] = Field(default_factory=list)


class CrystallizationGovernanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposed_by: str
    approved_by: str | None = None
    approval_mode: Literal["auto_policy", "operator", "manual_required"] = "manual_required"

    validation_status: Literal["unvalidated", "valid", "invalid", "quarantined"] = "unvalidated"
    validation_errors: list[str] = Field(default_factory=list)

    requires_manual_review: bool = True
    sensitivity: Literal["public", "private", "intimate"] = "private"

    created_from_policy: str | None = None
    last_reviewed_at: datetime | None = None


class CrystallizationProjectionRefsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_card_ids: list[str] = Field(default_factory=list)
    chroma_doc_ids: list[str] = Field(default_factory=list)
    rdf_named_graphs: list[str] = Field(default_factory=list)

    graphiti_episode_ids: list[str] = Field(default_factory=list)
    graphiti_entity_ids: list[str] = Field(default_factory=list)
    graphiti_edge_ids: list[str] = Field(default_factory=list)

    synced_at: datetime | None = None


class CrystallizationLinkV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_crystallization_id: str
    relation: CrystallizationRelation
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    note: str | None = None


class MemoryCrystallizationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["memory_crystallization.v1"] = "memory_crystallization.v1"

    crystallization_id: str
    kind: CrystallizationKind

    subject: str
    summary: str

    status: CrystallizationStatus = "proposed"
    confidence: CrystallizationConfidence = "likely"
    salience: float = Field(default=0.5, ge=0.0, le=1.0)

    scope: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    claims: list[CrystallizationClaimV1] = Field(default_factory=list)
    evidence: list[CrystallizationEvidenceRefV1] = Field(default_factory=list)

    source_card_ids: list[str] = Field(default_factory=list)
    source_grammar_event_ids: list[str] = Field(default_factory=list)
    source_atom_ids: list[str] = Field(default_factory=list)

    grammar_envelope: MemoryGrammarEnvelopeV1 = Field(default_factory=MemoryGrammarEnvelopeV1)

    planning_effects: list[str] = Field(default_factory=list)
    retrieval_affordances: list[str] = Field(default_factory=list)

    links: list[CrystallizationLinkV1] = Field(default_factory=list)
    projection_refs: CrystallizationProjectionRefsV1 = Field(
        default_factory=CrystallizationProjectionRefsV1
    )
    governance: CrystallizationGovernanceV1

    created_at: datetime
    updated_at: datetime


class ActiveMemoryPacketV1(BaseModel):
    """Compact, inspectable active memory context built by the retriever."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["active_memory_packet.v1"] = "active_memory_packet.v1"

    query: str
    task_type: str | None = None
    project_id: str | None = None
    session_id: str | None = None

    stance: list[dict[str, Any]] = Field(default_factory=list)
    project_state: list[dict[str, Any]] = Field(default_factory=list)
    procedures: list[dict[str, Any]] = Field(default_factory=list)
    open_loops: list[dict[str, Any]] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    attractors: list[dict[str, Any]] = Field(default_factory=list)

    card_refs: list[str] = Field(default_factory=list)
    crystallization_refs: list[str] = Field(default_factory=list)
    graphiti_refs: list[str] = Field(default_factory=list)
    chroma_refs: list[str] = Field(default_factory=list)
    rdf_refs: list[str] = Field(default_factory=list)

    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
