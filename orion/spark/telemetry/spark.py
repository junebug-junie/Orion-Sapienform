from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, Field

class SparkTelemetryPayload(BaseModel):
    """
    Telemetry payload for Spark/Tissue internal state updates.
    Corresponds to the 'spark.introspection.log' envelope kind.
    """
    correlation_id: UUID
    
    # Core Tissue Metrics
    phi: float         # Integrated Information / Coherence
    novelty: float     # Stimulus novelty (0.0 - 1.0)
    
    # Aggregation dimensions
    trace_mode: str
    trace_verb: str
    
    # Debug info
    stimulus_summary: str  # e.g. "v=0.8 a=0.5"
    
    # Full stats dump (valence, energy, etc.)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    timestamp: Optional[float] = None
