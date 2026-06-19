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
