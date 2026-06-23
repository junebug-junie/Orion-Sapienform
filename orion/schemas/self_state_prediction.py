from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SelfStatePredictionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["self_state_prediction.v1"] = "self_state_prediction.v1"
    prediction_id: str
    generated_at: datetime
    source_self_state_id: str
    predicted_dimension_scores: dict[str, float] = Field(default_factory=dict)
