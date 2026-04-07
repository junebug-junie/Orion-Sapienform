"""I/O contracts for reasoning artifact materialization (Phase 2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .reasoning import (
    ClaimV1,
    ConceptV1,
    ContradictionV1,
    MentorProposalV1,
    PromotionDecisionV1,
    ReasoningSparkStateSnapshotV1,
    RelationV1,
    VerbEvaluationV1,
)

ReasoningArtifactV1 = Annotated[
    Union[
        ClaimV1,
        ConceptV1,
        RelationV1,
        ContradictionV1,
        MentorProposalV1,
        PromotionDecisionV1,
        VerbEvaluationV1,
        ReasoningSparkStateSnapshotV1,
    ],
    Field(discriminator="artifact_type"),
]


class ReasoningWriteContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_family: Literal["concept_induction", "autonomy", "spark", "manual", "other"]
    source_kind: str
    source_channel: str
    producer: str
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None


class ReasoningWriteRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(default_factory=lambda: f"reasoning-write-{uuid4()}")
    context: ReasoningWriteContextV1
    artifacts: list[ReasoningArtifactV1] = Field(min_length=1)
    idempotency_key: Optional[str] = None
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("requested_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class ReasoningWriteResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str
    accepted: bool
    stored_count: int = 0
    deduped: bool = False
    artifact_ids: list[str] = Field(default_factory=list)
    status: Literal["stored", "deduped", "rejected"] = "stored"
    message: Optional[str] = None
