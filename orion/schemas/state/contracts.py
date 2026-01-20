from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.schemas.telemetry.biometrics import BiometricsSummaryV1, BiometricsInductionV1, BiometricsClusterV1


StateScope = Literal["global", "node"]
FreshnessStatus = Literal["fresh", "stale", "missing"]


class StateGetLatestRequest(BaseModel):
    """RPC request payload for the state read-model service."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    scope: StateScope = Field("global", description="global or per-node")
    node: Optional[str] = Field(None, description="required when scope='node'")
    kind: Literal["state.get_latest.v1"] = "state.get_latest.v1"


class StateLatestReply(BaseModel):
    """RPC reply payload from the state read-model service."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ok: bool = True

    status: FreshnessStatus
    as_of_ts: Optional[datetime] = None
    age_ms: Optional[int] = None
    kind: Literal["state.latest.reply.v1"] = "state.latest.reply.v1"

# The canonical snapshot when present
    snapshot: Optional[SparkStateSnapshotV1] = None

    # Diagnostics / hints (stable keys)
    source: Optional[str] = None
    note: Optional[str] = None
    biometrics: Optional["BiometricsContext"] = None


class BiometricsContext(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    status: FreshnessStatus
    age_ms: Optional[int] = None
    note: Optional[str] = None
    summary: Optional[BiometricsSummaryV1] = None
    induction: Optional[BiometricsInductionV1] = None
    cluster: Optional[BiometricsClusterV1] = None
