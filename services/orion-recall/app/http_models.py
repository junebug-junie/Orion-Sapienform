from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecallRequestBody(BaseModel):
    """HTTP API request model (backwards compatibility)."""

    text: str
    max_items: int = 8
    time_window_days: int = 90
    mode: str = "hybrid"
    tags: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = None
    phi: Optional[Dict[str, Any]] = None


class RecallResponseBody(BaseModel):
    fragments: List[Dict[str, Any]]
    debug: Dict[str, Any] = Field(default_factory=dict)
