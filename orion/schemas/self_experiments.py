"""Typed self-experiment registry schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SelfExperimentType = Literal[
    "skill_probe",
    "runtime_drift_check",
    "belief_origin_check",
    "trace_failure_autopsy",
    "repo_change_probe",
    "daily_focus_grounding_check",
    "memory_correction_candidate",
    "patch_proposal_candidate",
    "manual_review_candidate",
]

SelfExperimentSource = Literal[
    "daily_pulse_v1",
    "daily_metacog_v1",
    "manual",
    "journal",
    "world_pulse",
    "collapse_mirror",
    "scheduler",
]

SelfExperimentPriority = Literal["low", "normal", "high"]

SelfExperimentMutationPolicy = Literal["forbidden", "proposal_only", "dry_run_only"]

SelfExperimentStatus = Literal[
    "created",
    "validated",
    "rejected",
    "queued",
    "dispatching",
    "running",
    "completed",
    "failed",
    "blocked_for_evidence",
    "pending_review",
    "proposal_stored",
    "discarded",
    "expired",
]


class SelfExperimentSpecV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    experiment_type: SelfExperimentType
    question: str
    rationale: str | None = None

    source: SelfExperimentSource = "manual"
    source_ref: str | None = None
    correlation_id: str | None = None
    session_id: str = "orion_self_experiments"
    user_id: str = "juniper_primary"

    priority: SelfExperimentPriority = "normal"

    requested_skill_id: str | None = None
    requested_context_exec_mode: str | None = None

    scopes: dict[str, Any] = Field(default_factory=dict)
    args: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    mutation_policy: SelfExperimentMutationPolicy = "forbidden"

    created_at_utc: str


class SelfExperimentCreateRequestV1(BaseModel):
    """Accept legacy skill_id payloads or typed experiment fields."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str | None = Field(default=None, min_length=1, max_length=120)
    experiment_type: str | None = None
    question: str | None = Field(default=None, min_length=1, max_length=500)
    rationale: str | None = Field(default=None, max_length=500)
    source: SelfExperimentSource | None = None
    source_ref: str | None = None
    correlation_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    priority: SelfExperimentPriority = "normal"
    requested_skill_id: str | None = Field(default=None, min_length=1, max_length=120)
    requested_context_exec_mode: str | None = None
    scopes: dict[str, Any] = Field(default_factory=dict)
    args: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    mutation_policy: SelfExperimentMutationPolicy | None = None


class SelfExperimentCreateResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    experiment_id: str
    status: SelfExperimentStatus
    message: str | None = None


class SelfExperimentRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    spec: SelfExperimentSpecV1

    status: SelfExperimentStatus
    reason: str | None = None

    dedupe_key: str
    dispatch_attempts: int = 0

    context_exec_request: dict[str, Any] | None = None
    context_exec_run_id: str | None = None
    context_exec_status: str | None = None
    artifact_type: str | None = None
    artifact_summary: str | None = None
    artifact_payload: dict[str, Any] | None = None

    proposal_id: str | None = None
    proposal_status: str | None = None
    attention_required: bool = False

    created_at_utc: str
    updated_at_utc: str
    completed_at_utc: str | None = None


class SelfExperimentDispatchRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelfExperimentDispatchResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    experiment_id: str
    status: SelfExperimentStatus
    context_exec_mode: str | None = None
    expected_artifact_type: str | None = None
    message: str | None = None


class SelfExperimentListResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    total: int
    items: list[SelfExperimentRecordV1] = Field(default_factory=list)
