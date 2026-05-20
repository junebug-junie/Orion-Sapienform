from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ContextPackTargetApiV1(str, Enum):
    cursor = "cursor"
    codex = "codex"
    claude_code = "claude_code"
    orion = "orion"


class KnowledgeForgeStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "knowledge_forge_status.v1"
    generated_at: datetime
    repo_root: str
    source_count: int
    claim_count: int
    accepted_claim_count: int
    disputed_claim_count: int
    stale_claim_count: int
    spec_count: int
    execution_ready_spec_count: int
    decision_count: int
    pending_review_count: int
    context_pack_count: int
    warnings: list[str] = Field(default_factory=list)
    enabled: bool = True
    write_enabled: bool = False


class ClaimSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    statement: str
    status: str
    source_refs: list[str] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    used_by: list[str] = Field(default_factory=list)
    path: str | None = None


class SpecSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    title: str
    status: str
    component: str | None = None
    requirements: list[str]
    non_goals: list[str] = Field(default_factory=list)
    acceptance_tests: list[str] = Field(default_factory=list)
    source_claims: list[str] = Field(default_factory=list)
    likely_files: list[str] = Field(default_factory=list)
    known_traps: list[str] = Field(default_factory=list)
    path: str | None = None


class DecisionSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: str
    decision: str
    rationale: str
    path: str | None = None


class ReviewSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    target: str
    action: str
    path: str


class ContextPackSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    target: str
    task: str
    included_specs: list[str] = Field(default_factory=list)
    included_claim_ids: list[str] = Field(default_factory=list)
    path: str | None = None


class SourceSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    path: str
    trust_level: str


class ContextPackCompileRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    target: ContextPackTargetApiV1
    spec_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    include_disputed: bool = False
    include_stale: bool = False
    write_file: bool = False


class SearchHitV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    id: str
    label: str
    status: str | None = None
    path: str | None = None
    score: int = 0


class ContextPackCompileResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pack_id: str
    path: str | None = None
    target: str
    task: str
    included_specs: list[str] = Field(default_factory=list)
    included_claims: list[str] = Field(default_factory=list)
    excluded_claims: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    content: str


class IdeationMode(str, Enum):
    arsonist_review = "arsonist_review"
    spec_critique = "spec_critique"
    missing_questions = "missing_questions"
    context_pack_review = "context_pack_review"
    implementation_plan_review = "implementation_plan_review"


class IdeationRunRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(..., min_length=1)
    mode: IdeationMode = IdeationMode.arsonist_review
    target: str = "review_artifact"
    input_paths: list[str] = Field(default_factory=list)
    write_review: bool = False
    max_tokens: int = Field(4096, ge=256, le=16384)


class IdeationRunResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    provider: str
    model: str
    status: str
    summary: str
    content: str
    artifact_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    usage: dict[str, int | str] = Field(default_factory=dict)
