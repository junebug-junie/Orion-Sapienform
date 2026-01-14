from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetacogTriggerV1(BaseModel):
    trigger_kind: str = Field(..., description="baseline | dense | manual | pulse")
    reason: str
    zen_state: str = Field("unknown", description="zen | not_zen | unknown")
    pressure: float = Field(0.0, ge=0.0, le=1.0)
    window_sec: int = 15
    frame_refs: List[str] = Field(default_factory=list)
    signal_refs: List[str] = Field(default_factory=list)
    upstream: Dict[str, Any] = Field(default_factory=dict, description="Compact summary of upstream trigger event")
    recall_enabled: Optional[bool] = Field(
        default=None,
        description="Override for downstream recall usage (true/false), None to use defaults.",
    )
    timestamp: str = Field(default_factory=_utc_now_iso)
