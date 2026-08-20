from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ChatTurnStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: str
    turn_id: str
    session_id: str
    node_id: str
    observed_at: datetime
    word_count: int = 0
    repair_pressure_level: float = 0.0
    repair_pressure_confidence: float = 0.0
    has_repair_signal: bool = False
    stance_disposition: str = "unknown"
    stance_disposition_reasons: list[str] = Field(default_factory=list)
    stance_boundary_register: bool = False
    evidence_event_ids: list[str] = Field(default_factory=list)
    last_updated_at: datetime


class ChatSessionProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["chat_session_projection.v1"] = "chat_session_projection.v1"
    projection_id: str
    generated_at: datetime
    turns: dict[str, ChatTurnStateV1] = Field(default_factory=dict)
    total_turn_count: int = 0
    sessions: list[str] = Field(default_factory=list)
    # EWMA baseline of chat_prediction_error's raw per-tick mean pressure-hint
    # delta (orion/substrate/prediction_error.py), same shape and same reason as
    # ExecutionTrajectoryProjectionV1's identically-named fields: lets that
    # function score deviation from its own live baseline instead of the
    # module's fixed `_THRESHOLD = 0.30` divisor, which live-verified 2026-08-19
    # reads chat's real deltas (derived raw std ~7.2e-4 from a 7-day, 19,425-tick
    # window) as effectively always-near-zero after the /0.30 scale -- the same
    # disease execution_prediction_error was fixed for on 2026-07-28. Defaults
    # are the correct cold-start value for both a fresh projection and an
    # upgrade from a persisted-but-older row that predates these fields.
    prediction_error_baseline_ewma: float = 0.0
    prediction_error_baseline_ewma_var: float = 0.0
    prediction_error_baseline_ewma_n: int = 0
