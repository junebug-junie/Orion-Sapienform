from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class SparkTelemetryPayload(BaseModel):
    """Spark Engine telemetry written by orion-spark-introspector.

    Notes:
      - `phi` is kept as a float for backward compatibility with the existing
        DB schema (typically used to store coherence).
      - Rich, structured φ stats belong in `metadata["phi_stats"]` (and friends).
    """

    model_config = ConfigDict(extra="ignore")

    # NOTE: correlation_id belongs to the Titanium/BaseEnvelope. We accept an
    # optional copy here for backwards-compatibility, but consumers should use
    # envelope.correlation_id as source-of-truth.
    correlation_id: Optional[str] = None

    phi: float
    novelty: float

    trace_mode: str
    trace_verb: str
    stimulus_summary: str

    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
