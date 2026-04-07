"""Mentor gateway contracts for bounded external critique loop (Phase 6)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

MentorTaskTypeV1 = Literal[
    "ontology_cleanup",
    "contradiction_review",
    "concept_refinement",
    "autonomy_review",
    "missing_evidence_scan",
    "goal_critique",
    "verb_eval_review",
]


class MentorConstraintsV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    allow_identity_edits: bool = False
    allow_goal_commits: bool = False
    output_json_only: bool = True


class MentorContextSliceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    artifact_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    summary_refs: list[str] = Field(default_factory=list)


class MentorRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(default_factory=lambda: f"mentor-request-{uuid4()}")
    mentor_provider: str
    mentor_model: str
    task_type: MentorTaskTypeV1
    anchor_scope: Literal["orion", "juniper", "relationship", "world", "session"]
    subject_ref: Optional[str] = None
    context: MentorContextSliceV1 = Field(default_factory=MentorContextSliceV1)
    constraints: MentorConstraintsV1 = Field(default_factory=MentorConstraintsV1)
    correlation_id: Optional[str] = None
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("requested_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class MentorProposalItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    proposal_id: str
    proposal_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)
    suggested_payload: dict = Field(default_factory=dict)
    risk_tier: Literal["low", "medium", "high"] = "low"


class MentorResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    proposal_batch_id: str
    mentor_provider: str
    mentor_model: str
    task_type: MentorTaskTypeV1
    proposals: list[MentorProposalItemV1] = Field(default_factory=list)


class MentorGatewayResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str
    success: bool
    response: Optional[MentorResponseV1] = None
    failure_reason: Optional[str] = None
    materialized_count: int = 0
    materialized_artifact_ids: list[str] = Field(default_factory=list)
    audit: dict = Field(default_factory=dict)
