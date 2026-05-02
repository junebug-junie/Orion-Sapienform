from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ChatTurnRecord:
    turn_id: str
    correlation_id: str | None
    created_at: str
    prompt: str
    response: str
    thought_process: str
    source: str | None
    memory_status: str | None
    memory_tier: str | None
    memory_reason: str | None
    spark_meta: dict[str, Any]
    trace_mode: str | None
    trace_verb: str | None
    mode: str | None
    selected_ui_route: str | None
    thinking_source: str | None
    model: str | None
    has_thought_process: bool
    has_code: bool
    has_logs: bool
    has_error: bool
    has_commands: bool
    anchor_terms: list[str] = field(default_factory=list)


@dataclass
class AnchorRecord:
    turn_id: str
    anchors: list[str]
    anchor_types: list[str]
    surface_forms: list[str] = field(default_factory=list)


@dataclass
class GraphEdgeReason:
    reason_type: str
    reason_weight: float
    detail: str | None = None


@dataclass
class TurnGraphEdge:
    from_turn_id: str
    to_turn_id: str
    weight: float
    reasons: list[GraphEdgeReason]


@dataclass
class EpisodeRecord:
    episode_id: str
    start_at: str
    end_at: str
    turn_ids: list[str]
    top_anchors: list[str]
    confidence: float
    episode_label: str
    episode_summary: str


@dataclass
class TurnBlockRecord:
    turn_id: str
    created_at: str
    user_problem_block: str
    assistant_answer_block: str
    command_or_code_block: str
    log_or_error_block: str
    optional_reasoning_summary_block: str


@dataclass
class ClaimResolutionRecord:
    claim_id: str
    episode_id: str
    claim_text: str
    status: str
    resolution_text: str
    evidence_turn_ids: list[str]
    status_reason: str
    last_status_at: str


@dataclass
class PipelineManifest:
    pipeline_version: str
    run_id: str
    date_window_start: str
    date_window_end: str
    stage_stats: dict[str, dict[str, Any]]
    generated_at: str


def stable_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
