from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field


class RecallRequestBody(BaseModel):
    """HTTP API request model (backwards compatibility)."""

    # Accept both historical key "text" and newer key "query_text".
    query_text: str = Field(..., validation_alias=AliasChoices("text", "query_text"))

    # Optional override for which recall profile to use for this request.
    profile: Optional[str] = None

    # If true, include decision + backend counts in the HTTP response.
    diagnostic: bool = False

    max_items: int = 8
    time_window_days: int = 90
    mode: str = "hybrid"
    tags: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = None
    phi: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    packs: List[str] = Field(default_factory=list)


class RecallResponseBody(BaseModel):
    bundle: Dict[str, Any]
    debug: Dict[str, Any] = Field(default_factory=dict)


class RecallCompareRequestBody(BaseModel):
    query_text: str = Field(..., validation_alias=AliasChoices("text", "query_text"))
    profile: Optional[str] = None
    session_id: Optional[str] = None
    node_id: Optional[str] = None


class RecallCompareResponseBody(BaseModel):
    v1: Dict[str, Any]
    v2: Dict[str, Any]
    compare: Dict[str, Any]


class LivedKnowRequestBody(BaseModel):
    """``POST /lived/know``: a deliberate lookup (``python -P -m orion.dig know <words>``, #3535)."""

    words: str = Field(..., min_length=1, max_length=1000)
    # The chat turn's correlation id, so the lookup counts toward read demand for that turn.
    turn_id: Optional[str] = Field(default=None, max_length=200)
    max_hits: int = Field(default=8, ge=1, le=12)


class LivedKnowHit(BaseModel):
    strength: str  # "strong" (would be injected) | "weak" (below the bar, shares a word)
    kind: str
    purpose: str
    id: str
    date: str
    text: str
    fit: float
    # The day a memory's stated end date passed ("away till Friday"), else None. Such a hit
    # was true then, not now.
    ended: Optional[str] = None


class LivedKnowResponseBody(BaseModel):
    status: str
    backend: str
    degraded: List[str] = Field(default_factory=list)
    hits: List[LivedKnowHit] = Field(default_factory=list)
    unverified: int = 0  # matches found but not shown: the serving-gate re-check failed
    corr_id: str
