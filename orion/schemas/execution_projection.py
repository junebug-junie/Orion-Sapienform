from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionRunStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    correlation_id: str
    session_id: str | None = None
    turn_id: str | None = None
    node_id: str
    verb: str = "unknown"
    mode: str = "unknown"
    status: str = "unknown"
    step_count: int = 0
    started_step_count: int = 0
    completed_step_count: int = 0
    failed_step_count: int = 0
    recall_observed: bool = False
    final_text_present: bool = False
    reasoning_present: bool = False
    thinking_source: str = "none"
    pressure_hints: dict[str, float] = Field(default_factory=dict)
    evidence_event_ids: list[str] = Field(default_factory=list)
    last_updated_at: datetime


class ExecutionTrajectoryProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["projection.execution_trajectory.v1"] = (
        "projection.execution_trajectory.v1"
    )
    projection_id: str
    generated_at: datetime
    runs: dict[str, ExecutionRunStateV1] = Field(default_factory=dict)
