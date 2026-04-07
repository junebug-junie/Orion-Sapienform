"""Canonical spark source seam used by reasoning adapters (Phase 5)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SparkSourceSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_service: str
    source_node: Optional[str] = None
    snapshot_ts: datetime
    source_snapshot_id: str
    correlation_id: Optional[str] = None
    source_kind: str = "spark.source.snapshot.v1"

    dimensions: Dict[str, float] = Field(default_factory=dict)
    tensions: list[str] = Field(default_factory=list)
    attention_targets: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("snapshot_ts")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
