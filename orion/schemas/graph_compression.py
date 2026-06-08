from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class CompressionRegionV1(BaseModel):
    """A cached semantic compression of a graph region, written to orion:compressions."""

    region_id: str
    scope: Literal["episodic", "substrate", "self_study"]
    kind: Literal["community", "hotspot", "contradiction", "self_study_cluster"]
    summary: str
    summary_kind: Literal["llm", "structural"]
    salience: float
    trust_tier: str
    exemplar_ids: Annotated[list[str], Field(min_length=1)]
    derived_from: Annotated[list[str], Field(min_length=1)]
    generated_at: datetime
    compression_version: str
    stale: bool = False


class CompressionStalenessMarkV1(BaseModel):
    """Bus payload marking a graph region stale when source triples are written."""

    region_id: str | None = None
    scope: str | None = None
    reason: str
    source_service: str
    ts: float


class GraphCompressionRegionMaterializedV1(BaseModel):
    """Bus event emitted after each compression artifact is written to Fuseki."""

    region_id: str
    scope: str
    kind: str
    salience: float
    trust_tier: str
    summary_kind: str
    compression_version: str
    ts: float
