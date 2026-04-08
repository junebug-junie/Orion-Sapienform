from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ChatResponseFeedbackV1(BaseModel):
    """Operator/user feedback event for a chat response."""

    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(default_factory=lambda: f"chat-response-feedback-{uuid4()}")
    correlation_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    response_id: str | None = None
    rating: str = Field(default="neutral")
    feedback_text: str | None = None
    tags: list[str] = Field(default_factory=list, max_length=16)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
