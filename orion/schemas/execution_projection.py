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
    # cortex-exec-only step count, the mirror-image counterpart to
    # harness_started_step_count above -- tracked as its own independently
    # max()-merged counter, NOT derived as (started_step_count -
    # harness_started_step_count) at read time. A derived subtraction breaks under
    # merge.py's independent per-field max()-merge: cortex-exec and harness-governor
    # flush their grammar events separately, so a poll-bounded batch commonly contains
    # only one service's steps -- if a harness-heavy batch's started_step_count "wins"
    # the max() against an earlier cortex-exec-only batch's smaller total, the derived
    # subtraction silently reads 0 despite real cortex-exec steps having occurred (live
    # in code review). Tracking it directly avoids this entirely, mirroring
    # harness_started_step_count's own correct pattern. See NODE_CHANNELS
    # "execution_load".
    cortex_exec_started_step_count: int = 0
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
    # Real magnitude backing reasoning_present: char length of reasoning_content +
    # reasoning_trace at the point orion-cortex-exec's router.py already has both in
    # hand (record_assembled_grammar's call site). reasoning_present alone was a
    # boolean wearing a magnitude's name -- every turn that used any reasoning at all
    # read identically regardless of how much. See NODE_CHANNELS "reasoning_load".
    reasoning_char_count: int = 0
    # FCC-motor-only reasoning magnitude: output_tokens from the harness CLI's own
    # result-event usage object (real provider-computed tokens, not a char
    # approximation). Takes priority over reasoning_char_count in reasoning_load's
    # formula when present -- see grammar_extract.py. cortex-exec has no equivalent
    # field today; this is harness-governor-sourced only.
    reasoning_output_tokens: int = 0
    thinking_source: str = "none"
    # Deterministic tool-name-classified step composition for one FCC-motor turn:
    # how many tool_use calls were read-only research/context tools (Read, Grep,
    # WebSearch, mcp__gitnexus__*, mcp__firecrawl__*, ...) vs action/mutation tools
    # (Bash, Edit, Write, other MCP tools). A fixed allowlist match, NOT a taxonomy
    # service -- an unmatched tool name increments neither counter. See NODE_CHANNELS
    # "context_gathering_ratio".
    context_gathering_step_count: int = 0
    execution_step_count: int = 0
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
