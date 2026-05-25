from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionDispatchCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_id: str

    source_decision_id: str
    source_proposal_id: str

    dispatch_status: Literal[
        "prepared",
        "dry_run",
        "blocked",
        "dispatched",
        "skipped",
    ]

    dispatch_mode: Literal[
        "dry_run",
        "prepare_only",
        "dispatch_read_only",
    ]

    dispatch_kind: Literal[
        "inspect",
        "summarize",
        "observe",
        "noop",
    ]

    target_id: str
    target_kind: str

    cortex_verb: str | None = None
    cortex_mode: str | None = None

    request_envelope: dict[str, object] = Field(default_factory=dict)

    constraints: dict[str, str] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)

    risk_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)


class ExecutionDispatchFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["execution.dispatch.frame.v1"] = "execution.dispatch.frame.v1"

    frame_id: str
    generated_at: datetime

    source_policy_frame_id: str
    source_proposal_frame_id: str
    source_self_state_id: str

    execution_dispatch_policy_id: str = "execution_dispatch_policy.v1"

    dispatch_mode: Literal[
        "dry_run",
        "prepare_only",
        "dispatch_read_only",
    ] = "dry_run"

    candidates: list[ExecutionDispatchCandidateV1] = Field(default_factory=list)
    blocked_candidates: list[ExecutionDispatchCandidateV1] = Field(default_factory=list)
    dispatched_candidates: list[ExecutionDispatchCandidateV1] = Field(default_factory=list)

    dispatch_attempted: bool = False
    dispatch_count: int = 0
    blocked_count: int = 0

    warnings: list[str] = Field(default_factory=list)
