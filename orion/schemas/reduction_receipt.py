from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.state_delta import StateDeltaV1


class ProjectionUpdateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.projection_update.v1"] = "substrate.projection_update.v1"
    projection_kind: str
    projection_id: str
    node_id: str | None = None
    operation: str


class ReductionReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.reduction_receipt.v1"] = "substrate.reduction_receipt.v1"
    receipt_id: str
    emission_id: str | None = None
    organ_id: str | None = None
    accepted_event_ids: list[str] = Field(default_factory=list)
    rejected_event_ids: list[str] = Field(default_factory=list)
    merged_event_ids: list[str] = Field(default_factory=list)
    noop_event_ids: list[str] = Field(default_factory=list)
    state_deltas: list[StateDeltaV1] = Field(default_factory=list)
    projection_updates: list[ProjectionUpdateV1] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
