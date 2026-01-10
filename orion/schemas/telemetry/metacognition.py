# orion/schemas/telemetry/metacognition.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MetacognitionTickV1(BaseModel):
    tick_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: datetime

    source_service: str
    source_node: Optional[str] = None

    # “what happened”
    distress_score: Optional[float] = None
    zen_score: Optional[float] = None
    services_tracked: int = 0

    # freeform detail for now (keep it flexible, it’s telemetry)
    snapshot: Dict[str, Any] = Field(default_factory=dict)


class MetacognitionEnrichedV1(BaseModel):
    tick_id: str
    generated_at: datetime

    source_service: str
    source_node: Optional[str] = None

    distress_score: Optional[float] = None
    zen_score: Optional[float] = None
    services_tracked: int = 0

    tags: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)

    # keep original payload for audit/debug
    raw_tick: Dict[str, Any] = Field(default_factory=dict)
