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
