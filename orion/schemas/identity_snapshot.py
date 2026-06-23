from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IdentitySnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["identity_snapshot.v1"] = "identity_snapshot.v1"
    snapshot_id: str
    generated_at: datetime

    dominant_drive: str
    active_drives: list[str] = Field(default_factory=list)
    self_state_condition: str
    overall_intensity: float = Field(ge=0.0, le=1.0)
    summary_labels: list[str] = Field(default_factory=list)
    key_unknowns: list[str] = Field(default_factory=list)
    trajectory_condition: str = "unknown"
    source_self_state_id: str
    source_autonomy_subject: str = "orion"
