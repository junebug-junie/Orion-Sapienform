from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


CHAT_GPT_LOG_TURN_KIND = "chat.gpt.log.v1"
CHAT_GPT_MESSAGE_KIND = "chat.gpt.message.v1"


class ChatGptMessageV1(BaseModel):
    """ChatGPT imported message payload for isolated SQL + vector fanout."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    message_id: str
    session_id: Optional[str] = None
    role: str = "user"
    speaker: Optional[str] = None
    content: str
    timestamp: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class ChatGptLogTurnV1(BaseModel):
    """Turn-level ChatGPT import row (prompt + response) for `chat_gpt_log`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: Optional[str] = Field(default=None, description="Primary identifier for the turn row")
    correlation_id: Optional[str] = Field(default=None, description="Trace/correlation identifier")
    source: str = Field(..., description="Source label (e.g. chatgpt_import)")
    prompt: str = Field(..., description="User prompt")
    response: str = Field(..., description="Assistant response")
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    spark_meta: Optional[Dict[str, Any]] = Field(default=None)
