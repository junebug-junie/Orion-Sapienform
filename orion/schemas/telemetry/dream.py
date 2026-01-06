# orion/schemas/telemetry/dream.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class DreamRequest(BaseModel):
    """
    Payload logged on dream.log (what sql-writer validates).
    (Preserves prior behavior: extra ignored for forward-compat)
    """
    model_config = ConfigDict(extra="ignore")

    context_text: str
    mood: Optional[str] = None
    duration_seconds: int = 60
    integration_mode: str = "visual"  # "text", "visual", "audio"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DreamTriggerPayload(BaseModel):
    """Trigger payload for the dream service."""
    model_config = ConfigDict(extra="forbid")

    mode: str = Field("standard", description="Dream mode / profile")
