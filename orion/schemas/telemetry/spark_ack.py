from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SparkStateSnapshotAckV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ok: bool = True
    received_ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot_seq: Optional[int] = None
    snapshot_ts: Optional[datetime] = None
    note: Optional[str] = None
    source_service: str = "orion-state-service"
    source_node: Optional[str] = None

    @field_validator("received_ts", "snapshot_ts")
    @classmethod
    def _ensure_tz(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
