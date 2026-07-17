from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CausalGeometryEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    lag_sec: int
    strength: float = Field(ge=-1.0, le=1.0)  # signed correlation
    significance: float  # p-value or z-score from surrogate test, caller documents which
    n_samples: int = Field(ge=0)
    window_start: datetime
    window_end: datetime


class CausalGeometryDivergenceEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    observed_strength: float | None = None
    designed_weight: float | None = None
    delta: float | None = None  # observed - designed, when both present
    status: Literal["observed_only", "designed_only", "both", "insufficient_data"]


class CausalGeometrySnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["causal.geometry.snapshot.v1"] = "causal.geometry.snapshot.v1"
    snapshot_id: str
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    edges: list[CausalGeometryEdgeV1] = Field(default_factory=list)
    designed_topology_version: str | None = None
    divergence: list[CausalGeometryDivergenceEntryV1] = Field(default_factory=list)
    insufficient_data: bool = False
    notes: list[str] = Field(default_factory=list)


# Single source of truth for the `causal_geometry_snapshots` Postgres table's
# column list, shared by the writer (services/orion-sql-writer/app/models/
# causal_geometry_snapshot.py's CausalGeometrySnapshotSQL, whose column names
# match these exactly) and the reader (services/orion-hub/scripts/api_routes.py's
# SELECT). Both services already depend on this schemas package, so declaring it
# once here -- rather than in either service's own internals -- keeps them from
# drifting on column names/order without either reaching into the other's code.
# Deliberately excludes `schema_version`: it's a fixed literal, never a column.
CAUSAL_GEOMETRY_SNAPSHOT_SQL_COLUMNS: tuple[str, ...] = (
    "snapshot_id",
    "generated_at",
    "window_start",
    "window_end",
    "designed_topology_version",
    "insufficient_data",
    "edges",
    "divergence",
    "notes",
)
