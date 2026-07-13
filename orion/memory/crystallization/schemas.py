from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    "reflection",
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
    "autonomy_episode",
    # scripts/concept_relation_digest.py's evidence ref back onto the
    # memory_concept_relation_decisions row a "reflection" crystallization summarizes.
    "concept_relation_decision",
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

STANCE_PROCEDURE_DECISION_KINDS = frozenset({"stance", "procedure", "decision"})
CONTRADICTION_KIND = "contradiction"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


class CrystallizationRefV1(BaseModel):
    """Subschema ref embedded in projected MemoryCardV1 records."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["crystallization_ref.v1"] = "crystallization_ref.v1"
    crystallization_id: str
    kind: CrystallizationKind
    projection_role: Literal["recall_surface", "superseded_surface"] = "recall_surface"
    synced_at: datetime | None = None


class CrystallizationDynamicsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activation: float = Field(default=0.0, ge=0.0, le=1.0)
    reinforcement_count: int = 0
    formed_at: datetime | None = None
    last_reinforced_at: datetime | None = None
    last_recalled_at: datetime | None = None
    decay_half_life_days: float = 30.0
    retired_at: datetime | None = None


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
    dynamics: CrystallizationDynamicsV1 = Field(default_factory=CrystallizationDynamicsV1)

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
    projection_refs: CrystallizationProjectionRefsV1 = Field(default_factory=CrystallizationProjectionRefsV1)
    governance: CrystallizationGovernanceV1

    provenance: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime
    updated_at: datetime


class ActiveMemoryPacketV1(BaseModel):
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


class MemoryCrystallizationProposeRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: CrystallizationKind
    subject: str
    summary: str
    scope: list[str]
    confidence: CrystallizationConfidence = "likely"
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
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
    proposed_by: str = "operator"
    sensitivity: Literal["public", "private", "intimate"] = "private"

    @model_validator(mode="after")
    def _require_evidence(self) -> MemoryCrystallizationProposeRequestV1:
        if not self.evidence:
            raise ValueError("evidence is required for crystallization proposals")
        if not self.scope:
            raise ValueError("scope is required for crystallization proposals")
        return self

    def to_crystallization(self) -> MemoryCrystallizationV1:
        now = _utc_now()
        return MemoryCrystallizationV1(
            crystallization_id=new_crystallization_id(),
            kind=self.kind,
            subject=self.subject,
            summary=self.summary,
            status="proposed",
            confidence=self.confidence,
            salience=self.salience,
            scope=list(self.scope),
            tags=list(self.tags),
            claims=list(self.claims),
            evidence=list(self.evidence),
            source_card_ids=list(self.source_card_ids),
            source_grammar_event_ids=list(self.source_grammar_event_ids),
            source_atom_ids=list(self.source_atom_ids),
            grammar_envelope=self.grammar_envelope,
            planning_effects=list(self.planning_effects),
            retrieval_affordances=list(self.retrieval_affordances),
            links=list(self.links),
            governance=CrystallizationGovernanceV1(
                proposed_by=self.proposed_by,
                requires_manual_review=True,
                sensitivity=self.sensitivity,
            ),
            created_at=now,
            updated_at=now,
        )
