from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.journaler.schemas import JournalEntryWriteV1

TrustTier = Literal["authoritative", "induced", "reflective"]
SelfKnowledgeCategory = Literal[
    "service",
    "module",
    "channel",
    "verb",
    "schema",
    "touchpoint",
    "env_surface",
]
SelfWritebackState = Literal["written", "skipped", "failed"]
SelfConceptKind = Literal[
    "runtime_boundary",
    "service_cluster",
    "bus_topology_pattern",
    "recall_surface",
    "journaling_surface",
    "graph_surface",
]
SelfReflectionKind = Literal[
    "tension",
    "blind_spot",
    "growth_area",
    "capability_gap",
    "seam_risk",
    "followup_question",
    "candidate_skill_idea",
    "architecture_observation",
    "unmet_need",
    "recurring_constraint",
]
SelfStudyRetrievalMode = Literal["factual", "conceptual", "reflective"]
SelfStudyRecordType = Literal["fact", "concept", "reflection"]
SelfStudyStorageSurface = Literal["in_process", "rdf_graph", "journal"]
SelfStudyConsumerKind = Literal["delivery_debug", "planning_architecture", "metacog_self"]


class SelfKnowledgeItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    category: SelfKnowledgeCategory
    name: str
    trust_tier: TrustTier = "authoritative"
    observed_at: str
    run_id: str
    source_path: str
    origin_kind: str | None = None
    origin_name: str | None = None
    symbol_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelfKnowledgeSectionCountsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: int = 0
    modules: int = 0
    channels: int = 0
    verbs: int = 0
    schemas: int = 0
    touchpoints: int = 0
    env_surfaces: int = 0


class SelfSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    run_id: str
    observed_at: str
    repo_root: str
    trust_tier: TrustTier = "authoritative"
    counts: SelfKnowledgeSectionCountsV1
    services: list[SelfKnowledgeItemV1] = Field(default_factory=list)
    modules: list[SelfKnowledgeItemV1] = Field(default_factory=list)
    channels: list[SelfKnowledgeItemV1] = Field(default_factory=list)
    verbs: list[SelfKnowledgeItemV1] = Field(default_factory=list)
    schemas: list[SelfKnowledgeItemV1] = Field(default_factory=list)
    touchpoints: list[SelfKnowledgeItemV1] = Field(default_factory=list)
    env_surfaces: list[SelfKnowledgeItemV1] = Field(default_factory=list)


class SelfWritebackStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Literal["graph", "journal"]
    status: SelfWritebackState
    authoritative: bool
    channel: str | None = None
    graph: str | None = None
    idempotency_key: str | None = None
    append_only: bool = False
    detail: str | None = None


class SelfRepoInspectResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot: SelfSnapshotV1
    summary: str
    graph_write: SelfWritebackStatusV1
    journal_write: SelfWritebackStatusV1
    journal_entry: JournalEntryWriteV1


class SelfConceptEvidenceRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    item_id: str
    source_path: str
    origin_kind: str | None = None
    origin_name: str | None = None
    symbol_name: str | None = None
    trust_tier: Literal["authoritative"] = "authoritative"


class SelfInducedConceptV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_id: str
    concept_kind: SelfConceptKind
    label: str
    description: str
    trust_tier: Literal["induced"] = "induced"
    confidence: float
    source_snapshot_id: str
    evidence: list[SelfConceptEvidenceRefV1] = Field(default_factory=list)
    inferred_from: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelfConceptInduceResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source_snapshot_id: str
    concepts: list[SelfInducedConceptV1] = Field(default_factory=list)
    summary: str
    graph_write: SelfWritebackStatusV1


class SelfConceptRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_id: str
    concept_kind: SelfConceptKind
    label: str
    source_snapshot_id: str
    trust_tier: Literal["induced"] = "induced"


class SelfReflectiveFindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reflection_id: str
    reflection_kind: SelfReflectionKind
    title: str
    description: str
    trust_tier: Literal["reflective"] = "reflective"
    confidence: float
    salience: float
    source_snapshot_id: str
    evidence: list[SelfConceptEvidenceRefV1] = Field(default_factory=list)
    concept_refs: list[SelfConceptRefV1] = Field(default_factory=list)
    recommendation: str | None = None
    follow_up_question: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelfConceptReflectResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source_snapshot_id: str
    source_concept_ids: list[str] = Field(default_factory=list)
    validated_phase2a: bool = True
    validation_summary: str
    findings: list[SelfReflectiveFindingV1] = Field(default_factory=list)
    summary: str
    graph_write: SelfWritebackStatusV1
    journal_write: SelfWritebackStatusV1
    journal_entry: JournalEntryWriteV1


class SelfStudyRetrieveFiltersV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trust_tiers: list[TrustTier] = Field(default_factory=list)
    record_types: list[SelfStudyRecordType] = Field(default_factory=list)
    stable_ids: list[str] = Field(default_factory=list)
    concept_kinds: list[SelfConceptKind] = Field(default_factory=list)
    reflection_kinds: list[SelfReflectionKind] = Field(default_factory=list)
    source_kinds: list[str] = Field(default_factory=list)
    storage_surfaces: list[SelfStudyStorageSurface] = Field(default_factory=list)
    text_query: str | None = None
    limit: int = 12


class SelfStudyRetrieveRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_mode: SelfStudyRetrievalMode
    filters: SelfStudyRetrieveFiltersV1 = Field(default_factory=SelfStudyRetrieveFiltersV1)


class SelfStudyRetrievedRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stable_id: str
    trust_tier: TrustTier
    record_type: SelfStudyRecordType
    title: str
    content_preview: str
    source_kind: str
    storage_surface: SelfStudyStorageSurface = "in_process"
    source_snapshot_id: str
    source_path: str | None = None
    origin_kind: str | None = None
    origin_name: str | None = None
    symbol_name: str | None = None
    concept_kind: SelfConceptKind | None = None
    reflection_kind: SelfReflectionKind | None = None
    evidence: list[SelfConceptEvidenceRefV1] = Field(default_factory=list)
    concept_refs: list[SelfConceptRefV1] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelfStudyRetrievalGroupV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trust_tier: TrustTier
    items: list[SelfStudyRetrievedRecordV1] = Field(default_factory=list)


class SelfStudyRetrievalCountsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = 0
    authoritative: int = 0
    induced: int = 0
    reflective: int = 0
    facts: int = 0
    concepts: int = 0
    reflections: int = 0


class SelfStudyRetrievalBackendStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage_surface: SelfStudyStorageSurface
    status: Literal["used", "not_queried", "unavailable"]
    detail: str | None = None


class SelfStudyRetrieveResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    retrieval_mode: SelfStudyRetrievalMode
    applied_filters: SelfStudyRetrieveFiltersV1
    groups: list[SelfStudyRetrievalGroupV1] = Field(default_factory=list)
    counts: SelfStudyRetrievalCountsV1
    backend_status: list[SelfStudyRetrievalBackendStatusV1] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SelfStudyConsumerPolicyDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consumer_name: str
    consumer_kind: SelfStudyConsumerKind
    explicit: bool = False
    enabled: bool = False
    retrieval_mode: SelfStudyRetrievalMode | None = None
    max_mode: SelfStudyRetrievalMode | None = None
    allowed_trust_tiers: list[TrustTier] = Field(default_factory=list)
    policy_reason: str
    downgraded: bool = False


class SelfStudyConsumerContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consumer_name: str
    consumer_kind: SelfStudyConsumerKind
    retrieval_mode: SelfStudyRetrievalMode
    policy_reason: str
    used: bool = False
    result: SelfStudyRetrieveResultV1 | None = None
    notes: list[str] = Field(default_factory=list)
