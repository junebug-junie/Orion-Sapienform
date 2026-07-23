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
    # harness-governor-only step count, split from started_step_count (which blends
    # cortex-exec + harness-governor) so the FCC motor's own step load can be measured
    # without cortex-exec sub-steps diluting it. See NODE_CHANNELS "harness_step_load".
    harness_started_step_count: int = 0
    # HarnessRunV1.compliance_verdict threaded through as a grammar-stream kv; "unknown"
    # until a real exec_result_assembled event sets it.
    compliance_verdict: str = "unknown"
    # Per-turn FCC step verbosity: total/max chars across measure_step_payload_chars()
    # calls in the step loop. Average is computed at read time (step_char_sum /
    # completed_step_count) rather than stored, to avoid a second counter desyncing
    # from completed_step_count under independent max()-merge.
    step_char_sum: int = 0
    step_char_max: int = 0
    # Longest run of consecutive identical short_error_kind() buckets seen across
    # tool_result is_error blocks in one turn -- a stuck-loop proxy. Only the max is
    # kept (not a growing list of every failure) since this pipeline only ever sees a
    # single end-of-run flush per turn (see HarnessGrammarCollector), so "current
    # streak" and "max streak" are the same observation in practice.
    tool_failure_streak_max: int = 0
    # Set only by an exec_turn_timeout event (orion-hub-sourced, published from
    # orion/hub/turn_orchestrator.py when a harness-governor RPC never returns). This
    # run's other fields will otherwise be entirely at their defaults -- there is no
    # governor-side data for a turn that never reached the motor's own lifecycle flush.
    turn_timed_out: bool = False
    recall_observed: bool = False
    final_text_present: bool = False
    reasoning_present: bool = False
    thinking_source: str = "none"
    llm_serving_node: str | None = None
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
