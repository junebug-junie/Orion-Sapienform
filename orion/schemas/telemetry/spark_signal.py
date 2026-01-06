from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SparkSignalV1(BaseModel):
    """
    Normalized signal contract for Spark.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    signal_type: Literal["equilibrium", "recall", "routing", "resource", "human", "vision", "collapse"]
    intensity: float = Field(0.0, ge=0.0, le=1.0)
    valence_delta: Optional[float] = None
    arousal_delta: Optional[float] = None
    coherence_delta: Optional[float] = None
    novelty_delta: Optional[float] = None
    as_of_ts: datetime
    ttl_ms: int = 15000
    source_service: str
    source_node: Optional[str] = None

    @field_validator("as_of_ts")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
