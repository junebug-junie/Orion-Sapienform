from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class MemoryItemV1(BaseModel):
    """A single retrieved memory item."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(..., description="Stable identifier for the item")
    source: str = Field(..., description="Backend source (vector/rdf/sql/etc)")
    source_ref: Optional[str] = Field(None, description="Collection/index/table reference")
    uri: Optional[str] = Field(None, description="Canonical URI if available")
    score: float = Field(0.0, description="Normalized [0,1] relevance score")
    ts: Optional[float] = Field(None, description="Epoch seconds for recency ordering")
    title: Optional[str] = None
    snippet: str = Field("", description="Short text fragment")
    tags: List[str] = Field(default_factory=list)


class MemoryBundleStatsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_counts: Dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    profile: Optional[str] = None
    diagnostic: Optional[Dict[str, Any]] = None


class MemoryBundleV1(BaseModel):
    """Prompt-ready bundle produced by recall."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    rendered: str = Field("", description="Concise prompt-ready text")
    items: List[MemoryItemV1] = Field(default_factory=list)
    stats: MemoryBundleStatsV1 = Field(default_factory=MemoryBundleStatsV1)


class RecallQueryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fragment: str = Field(..., description="User fragment/query text")
    verb: Optional[str] = None
    intent: Optional[str] = None
    session_id: Optional[str] = None
    node_id: Optional[str] = None
    profile: str = Field("reflect.v1", description="Recall profile name")
    lane: Optional[str] = Field(
        default=None,
        description="Visibility lane selector (e.g. chat vs social) for memory cards scope.",
    )
    profile_explicit: bool = Field(
        default=False,
        description="When True, intent router should not override the requested recall profile.",
    )
    exclude: Optional[Dict[str, object]] = Field(
        default=None,
        description="Optional current-turn exclusion hints (ids/text/timestamps).",
    )
    reply_to: Optional[str] = None


class RecallReplyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    bundle: MemoryBundleV1
    correlation_id: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


class RecallDecisionV1(BaseModel):
    """Telemetry for recall decisions."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    corr_id: str
    session_id: Optional[str] = None
    node_id: Optional[str] = None
    verb: Optional[str] = None
    profile: Optional[str] = None
    query: str
    selected_ids: List[str] = Field(default_factory=list)
    dropped: Dict[str, int] = Field(default_factory=dict)
    backend_counts: Dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    recall_debug: Dict[str, Any] = Field(
        default_factory=dict,
        description="Bounded, structured diagnostics for recall decision explainability.",
    )
    ranking_debug: List[Dict[str, Optional[float | int | str | bool]]] = Field(
        default_factory=list,
        description="Optional relevance diagnostics for ranked candidates.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
