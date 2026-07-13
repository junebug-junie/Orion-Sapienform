from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionDispatchCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_id: str

    source_decision_id: str
    source_proposal_id: str

    dispatch_status: Literal[
        "prepared",
        "dry_run",
        "blocked",
        "prepared_for_dispatch",
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

    result_ref: str | None = None
    dispatch_error: str | None = None
    dispatched_at: datetime | None = None

    @model_validator(mode="after")
    def _dispatched_requires_evidence(self) -> "ExecutionDispatchCandidateV1":
        if self.dispatch_status == "dispatched":
            if self.dispatched_at is None or (self.result_ref is None and self.dispatch_error is None):
                raise ValueError(
                    "dispatch_status='dispatched' requires dispatched_at and one of "
                    "result_ref/dispatch_error as evidence a send was actually attempted; "
                    "use 'prepared_for_dispatch' for a candidate that has not been sent yet"
                )
        return self


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
