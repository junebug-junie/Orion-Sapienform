"""Schemas for time-bounded chat discussion extraction (chat_history_log)."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DiscussionWindowRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lookback_seconds: int = Field(..., ge=1, le=86400 * 14)
    end_time_utc: Optional[datetime] = None
    user_id: Optional[str] = None
    source: Optional[str] = None
    max_turns: int = Field(30, ge=1, le=200)
    require_prompt_and_response: bool = True


class DiscussionWindowTurnV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    source: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    prompt: str
    response: str


class DiscussionWindowResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_start_utc: datetime
    window_end_utc: datetime
    turn_count: int
    source: Optional[str] = None
    user_id: Optional[str] = None
    turns: List[DiscussionWindowTurnV1] = Field(default_factory=list)
    transcript_text: str = ""
    selection_strategy: str = "time_bound_then_contiguous_suffix"
