from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SparkConceptProfileGraphMaterializationV1(BaseModel):
    """Typed payload describing a Spark ConceptProfile graph materialization write."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_kind: Literal["spark_concept_profile"] = "spark_concept_profile"
    schema_kind: Literal["spark.concept_profile.graph.v1"] = "spark.concept_profile.graph.v1"
    profile_id: str
    subject: str
    revision: int = Field(ge=1)
    produced_at: datetime
    window_start: datetime
    window_end: datetime
    concept_count: int = Field(ge=0)
    cluster_count: int = Field(ge=0)
    state_estimate_present: bool
    correlation_id: str | None = None
    writer_service: str
    writer_version: str
