from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClaimStatusV1(str, Enum):
    accepted = "accepted"
    disputed = "disputed"
    stale = "stale"
    superseded = "superseded"
    speculative = "speculative"


class SpecStatusV1(str, Enum):
    draft = "draft"
    reviewed = "reviewed"
    execution_ready = "execution_ready"
    implemented = "implemented"
    stale = "stale"


class DecisionStatusV1(str, Enum):
    proposed = "proposed"
    accepted = "accepted"
    superseded = "superseded"


class ContextPackTargetV1(str, Enum):
    cursor = "cursor"
    codex = "codex"
    orion_agent = "orion-agent"
    human = "human"


class SourceKindV1(str, Enum):
    conversation = "conversation"
    paper = "paper"
    code = "code"
    issue = "issue"
    design_doc = "design_doc"


class TrustLevelV1(str, Enum):
    primary = "primary"
    secondary = "secondary"
    speculative = "speculative"


class TypedRelationsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    implements: list[str] = Field(default_factory=list)
    tested_by: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    motivated_by: list[str] = Field(default_factory=list)


class SourceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["source"]
    id: str
    kind: SourceKindV1
    path: str
    trust_level: TrustLevelV1 = TrustLevelV1.primary


class ClaimV1(TypedRelationsV1):
    model_config = ConfigDict(extra="forbid")

    type: Literal["claim"]
    id: str
    statement: str
    status: ClaimStatusV1
    source_refs: list[str]
    confidence: Literal["high", "medium", "low"] = "medium"
    used_by: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def claim_id_prefix(cls, value: str) -> str:
        if not value.startswith("claim:"):
            raise ValueError("claim id must start with claim:")
        return value


class DecisionV1(TypedRelationsV1):
    model_config = ConfigDict(extra="forbid")

    type: Literal["decision"]
    id: str
    status: DecisionStatusV1
    decision: str
    rationale: str
    consequences: list[str] = Field(default_factory=list)
    source_claims: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def decision_id_prefix(cls, value: str) -> str:
        if not value.startswith("adr:"):
            raise ValueError("decision id must start with adr:")
        return value


class SpecV1(TypedRelationsV1):
    model_config = ConfigDict(extra="forbid")

    type: Literal["spec"]
    id: str
    status: SpecStatusV1
    component: str
    requirements: list[str]
    non_goals: list[str] = Field(default_factory=list)
    acceptance_tests: list[str] = Field(default_factory=list)
    source_claims: list[str] = Field(default_factory=list)
    likely_files: list[str] = Field(default_factory=list)
    known_traps: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def spec_id_prefix(cls, value: str) -> str:
        if not value.startswith("spec:"):
            raise ValueError("spec id must start with spec:")
        return value

    @model_validator(mode="after")
    def execution_ready_needs_tests(self) -> SpecV1:
        if self.status == SpecStatusV1.execution_ready and not self.acceptance_tests:
            raise ValueError("execution_ready specs require acceptance_tests")
        return self


class ContextPackV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["context_pack"]
    id: str
    target: ContextPackTargetV1
    task: str
    allowed_sources: list[str] = Field(default_factory=list)
    included_specs: list[str] = Field(default_factory=list)
    excluded_context: list[str] = Field(default_factory=list)
    included_claim_ids: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def context_pack_id_prefix(cls, value: str) -> str:
        if not value.startswith("ctx:"):
            raise ValueError("context pack id must start with ctx:")
        return value
