from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MEMORY_TURN_PERSISTED_KIND = "memory.turn.persisted.v1"
CHAT_HISTORY_SPARK_META_PATCH_KIND = "chat.history.spark_meta.patch.v1"


class MemoryTurnPersistedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    prompt: str
    response: str
    spark_meta: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None


class ChatHistorySparkMetaPatchV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    spark_meta: Dict[str, Any] = Field(default_factory=dict)


class MemoryConsolidationWindowV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_window_id: str
    turn_correlation_ids: List[str] = Field(default_factory=list)
    status: Literal["open", "closed", "consolidated", "failed"] = "open"
    phase_change_at_close: Optional[str] = None
    consolidation_status: Optional[Literal["pending", "ok", "failed"]] = None
    draft_id: Optional[str] = None
    created_at: datetime
    closed_at: Optional[datetime] = None


class MemoryGraphSuggestDraftRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str
    memory_window_id: str
    status: Literal["pending_review", "approved", "rejected"] = "pending_review"
    draft: Dict[str, Any]
    turn_correlation_ids: List[str] = Field(default_factory=list)
    created_at: datetime
