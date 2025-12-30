from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecallRequestBody(BaseModel):
    """HTTP API request model (backwards compatibility)."""

    query_text: str = Field(..., alias="text")
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
    fragments: List[Dict[str, Any]]
    debug: Dict[str, Any] = Field(default_factory=dict)
