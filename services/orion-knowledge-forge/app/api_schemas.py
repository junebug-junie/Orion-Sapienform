from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ContextPackTargetApiV1(str, Enum):
    cursor = "cursor"
    codex = "codex"
    claude_code = "claude_code"
    orion = "orion"


class KnowledgeForgeStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    enabled: bool
    write_enabled: bool
    repo_root: str
    counts: dict[str, int]
    warnings: list[str] = Field(default_factory=list)


class ClaimSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    status: str
    confidence: str
    path: str | None = None


class SpecSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: str
    component: str
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


class ContextPackCompileResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    path: str | None = None
    warnings: list[str] = Field(default_factory=list)
