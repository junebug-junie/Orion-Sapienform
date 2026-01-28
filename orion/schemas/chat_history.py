from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from orion.core.bus.bus_schemas import Envelope, ServiceRef

CHAT_HISTORY_MESSAGE_KIND = "chat.history.message.v1"
CHAT_HISTORY_TURN_KIND = "chat.history"
ChatRole = Literal["user", "assistant", "system", "tool"]
MemoryStatus = Literal["accepted", "rejected", "pending"]
MemoryTier = Literal["durable", "ephemeral"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatHistoryMessageV1(BaseModel):
    """
    Canonical chat history payload for vector + SQL persistence.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    message_id: str = Field(
        default_factory=lambda: str(uuid4()),
        validation_alias=AliasChoices("message_id", "id"),
    )
    session_id: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("session_id", "conversation_id")
    )
    role: ChatRole = Field(default="user")
    speaker: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("speaker", "user", "author")
    )
    content: str
    timestamp: str = Field(default_factory=_now_iso)

    # Optional provenance
    model: Optional[str] = None
    provider: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    memory_status: Optional[MemoryStatus] = None
    memory_tier: Optional[MemoryTier] = None
    memory_reason: Optional[str] = None
    client_meta: Optional[Dict[str, Any]] = None

    def to_document(
        self,
        *,
        kind: str = CHAT_HISTORY_MESSAGE_KIND,
        source: Optional[ServiceRef] = None,
        correlation_id: Optional[UUID] = None,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert to a vector-writer friendly document.

        The caller supplies envelope context (kind, source, correlation_id, channel)
        so metadata can capture lineage without revalidating the envelope here.
        """
        metadata: Dict[str, Any] = {
            "message_id": self.message_id,
            "role": self.role,
            "speaker": self.speaker,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "kind": kind,
        }

        if correlation_id:
            metadata["correlation_id"] = str(correlation_id)
        if channel:
            metadata["source_channel"] = channel
        if source:
            metadata["source_service"] = source.name
            metadata["source_node"] = source.node
            metadata["source_version"] = source.version
            metadata["source_instance"] = source.instance
        if self.model:
            metadata["model"] = self.model
        if self.provider:
            metadata["provider"] = self.provider
        if self.tags:
            metadata["tags"] = self.tags
        if self.memory_status:
            metadata["memory_status"] = self.memory_status
        if self.memory_tier:
            metadata["memory_tier"] = self.memory_tier
        if self.memory_reason:
            metadata["memory_reason"] = self.memory_reason
        if self.client_meta:
            metadata["client_meta"] = self.client_meta

        # Drop None values for cleaner metadata blobs
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return {
            "id": self.message_id,
            "text": self.content,
            "metadata": metadata,
        }


class ChatHistoryMessageEnvelope(Envelope[ChatHistoryMessageV1]):
    """
    Versioned Titanium envelope for chat history log entries.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal[CHAT_HISTORY_MESSAGE_KIND] = Field(CHAT_HISTORY_MESSAGE_KIND)
    payload: ChatHistoryMessageV1

class ChatHistoryTurnV1(BaseModel):
    """Turn-level chat history row (prompt + response) for `chat_history_log`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: Optional[str] = Field(default=None, description="Primary identifier for the turn row")
    correlation_id: Optional[str] = Field(default=None, description="Trace/correlation identifier")
    source: str = Field(..., description="Source label (e.g. hub_ws)")
    prompt: str = Field(..., description="User prompt")
    response: str = Field(..., description="Assistant response")
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    spark_meta: Optional[Dict[str, Any]] = Field(default=None)
    memory_status: Optional[MemoryStatus] = None
    memory_tier: Optional[MemoryTier] = None
    memory_reason: Optional[str] = None
    client_meta: Optional[Dict[str, Any]] = None


class ChatHistoryTurnEnvelope(Envelope[ChatHistoryTurnV1]):
    """Versioned Titanium envelope for turn-level chat history logging."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal[CHAT_HISTORY_TURN_KIND] = Field(CHAT_HISTORY_TURN_KIND)
    payload: ChatHistoryTurnV1
