from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CabinetAmbientSpikeV1(BaseModel):
    """Bus contract when cabinet ambient audio activity crosses a sustained threshold.

    Producer: orion-biometrics (agent mode, per-node). v1 emits the event only;
    optional STT clip capture is a separate consumer path (deferred).
    """

    model_config = ConfigDict(extra="forbid")

    spike_id: str = Field(description="Unique id for this spike emission (trace dedup).")
    node: str
    timestamp: datetime
    activity: float = Field(ge=0.0, le=1.0, description="cabinet_ambient_audio_activity at fire time.")
    rms: float = Field(ge=0.0, description="cabinet_ambient_rms at fire time (PCM units).")
    peak: Optional[float] = Field(default=None, ge=0.0)
    activity_threshold: float = Field(ge=0.0, le=1.0)
    consecutive_ticks: int = Field(ge=1)
    source_service: str
    source_node: Optional[str] = None

    @field_validator("timestamp")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
