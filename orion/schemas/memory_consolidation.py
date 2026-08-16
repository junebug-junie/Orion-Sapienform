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
    # External-world provenance for this turn, read from the chat_history_log row's
    # client_meta.external_room.platform (e.g. "aitown"). None means a direct
    # hub/API conversation with Juniper -- the correct default, and what every row
    # predating this field carries. Downstream (formation_policy) uses this to keep
    # NPC dialogue out of the human review queue without discarding it as memory.
    #
    # DEPLOY ORDER MATTERS: this model is extra="forbid", so a consumer running
    # pre-#1672 code hard-fails validation on an envelope carrying this field.
    # orion-memory-consolidation must be deployed BEFORE orion-sql-writer.
    source_platform: Optional[str] = None


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
    consolidation_status: Optional[Literal["pending", "ok", "failed", "skipped"]] = None
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
