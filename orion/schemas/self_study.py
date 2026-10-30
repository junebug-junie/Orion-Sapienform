from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.journaler.schemas import JournalEntryWriteV1

# Layer 1 facts are the only self-study output left (2026-10-30 fold into curiosity,
# docs/superpowers/specs/2026-10-29-self-study-into-curiosity-proposal.md). The induced and
# reflective tiers belonged to the retired concept induction and Layer 3 reflection.
TrustTier = Literal["authoritative"]
SelfKnowledgeCategory = Literal[
    "service",
    "module",
    "channel",
    "verb",
    "schema",
    "touchpoint",
    "env_surface",
    # Added 2026-09-05 (Layer 1 broadening, self-model rebuild arc):
    # physical/hardware facts (config/field/orion_field_topology.v1.yaml) and
    # Orion's own aggregated conversational behavior (chat_stance_belief_log,
    # previously computed every turn and discarded -- see
    # services/orion-cortex-exec/app/self_study.py _hardware_items/
    # _behavioral_items).
    "hardware",
    "behavioral",
]
SelfWritebackState = Literal["written", "skipped", "failed"]
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
    hardware: int = 0
    behavioral: int = 0


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
    hardware: list[SelfKnowledgeItemV1] = Field(default_factory=list)
    behavioral: list[SelfKnowledgeItemV1] = Field(default_factory=list)


class SelfWritebackStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "self_knowledge_items" is the per-item write that feeds Self Atlas; "journal" is the one
    # daily summary entry. The RDF "graph" target and Layer 3's "self_concept_history" target
    # were retired 2026-10-30 with the code that produced them.
    target: Literal["journal", "self_knowledge_items"]
    status: SelfWritebackState
    authoritative: bool
    channel: str | None = None
    idempotency_key: str | None = None
    append_only: bool = False
    detail: str | None = None


class SelfRepoInspectResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot: SelfSnapshotV1
    summary: str
    journal_write: SelfWritebackStatusV1
    journal_entry: JournalEntryWriteV1
    # Added 2026-09-05 (self-model rebuild arc, Patch 2): review finding --
    # this was previously computed and discarded, so a run where every item
    # publish failed was indistinguishable from a fully successful one.
    self_knowledge_items_write: SelfWritebackStatusV1
