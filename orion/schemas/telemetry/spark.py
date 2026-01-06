from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class SparkStateSnapshotV1(BaseModel):
    """Canonical, versioned snapshot of Orion's internal state.

    This schema is used:
      - on the Titanium bus (real-time)
      - stored durably (embedded under spark_telemetry.metadata)
      - cached in Redis by orion-state-service

    It is designed to be restart-proof and ordering-safe.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # provenance
    source_service: str
    source_node: Optional[str] = None

    # restart + ordering
    producer_boot_id: str
    seq: int

    # time semantics
    snapshot_ts: datetime
    valid_for_ms: int = 15000

    # trace linkage
    correlation_id: Optional[str] = None
    trace_mode: Optional[str] = None
    trace_verb: Optional[str] = None

    # metrics
    phi: Dict[str, float] = Field(default_factory=dict, description="phi components (valence/energy/coherence/novelty, etc)")
    valence: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5

    vector_present: bool = False
    vector_ref: Optional[str] = None

    # optional extra
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def idempotency_key(self) -> str:
        return f"{self.producer_boot_id}:{self.seq}"


class SparkTelemetryPayload(BaseModel):
    """Spark telemetry row that maps to spark_telemetry table.

    Note: The durable table has a compact column set.
    The full canonical snapshot lives inside `metadata[\"spark_state_snapshot\"]`.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # columns
    telemetry_id: Optional[str] = None
    correlation_id: str
    phi: Optional[float] = None
    novelty: Optional[float] = None
    trace_mode: Optional[str] = None
    trace_verb: Optional[str] = None
    stimulus_summary: Optional[str] = None
    timestamp: datetime | str

    # json column
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # canonical snapshot (typed)
    state_snapshot: Optional[SparkStateSnapshotV1] = None
