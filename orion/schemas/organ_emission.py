from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.grammar import GrammarEventV1


class OrganEmissionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["organ.emission.v1"] = "organ.emission.v1"
    emission_id: str
    organ_id: str
    invocation_id: str
    triggered_by_event_ids: list[str] = Field(default_factory=list)
    inspected_projection_ids: list[str] = Field(default_factory=list)
    candidate_events: list[GrammarEventV1] = Field(default_factory=list)
    debug_trace: dict[str, Any] | None = None
    created_at: datetime
