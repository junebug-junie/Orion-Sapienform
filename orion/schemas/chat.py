from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

class RawChat(BaseModel):
    """
    Minimal payload capturing a raw chat exchange.
    """
    model_config = ConfigDict(extra="ignore")

    id: str
    session_id: str
    user_id: Optional[str] = None
    role: str # "user" or "assistant"
    content: str
    model: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatResultPayload(BaseModel):
    """
    Standardized payload for LLM generation results.
    """
    model_config = ConfigDict(extra="ignore")

    content: str
    role: str = "assistant"
    spark_vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EnrichedChat(BaseModel):
    """
    Enriched metadata for a chat message (tags, summary, embeddings pointer).
    """
    model_config = ConfigDict(extra="ignore")

    ref_id: str # Links to RawChat.id
    tags: List[str] = Field(default_factory=list)
    entities: List[Dict[str, str]] = Field(default_factory=list)
    sentiment: Optional[float] = None
    summary: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
