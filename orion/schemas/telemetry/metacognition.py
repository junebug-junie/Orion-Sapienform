# orion/schemas/telemetry/metacognition.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MetacognitionTickV1(BaseModel):
    tick_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    generated_at: datetime

    source_service: str
    source_node: Optional[str] = None

    # “what happened”
    distress_score: float = 0.0
    zen_score: float = 1.0
    services_tracked: int = 0

    # freeform detail for now (keep it flexible, it’s telemetry)
    snapshot: Dict[str, Any] = Field(default_factory=dict)

