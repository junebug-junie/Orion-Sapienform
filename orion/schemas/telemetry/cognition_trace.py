from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.cortex.schemas import StepExecutionResult


class CognitionTracePayload(BaseModel):
    """A canonical trace of a cognition run.

    This is the payload published on the bus and consumed by:
      - sql-writer (persistence)
      - spark-introspector (metacognition)
      - rdf-writer / vector-writer (memory ingestion)
    """

    model_config = ConfigDict(extra="ignore")

    # NOTE: correlation_id belongs to the Titanium/BaseEnvelope. We accept an
    # optional copy here for backwards-compatibility, but consumers should use
    # envelope.correlation_id as source-of-truth.
    correlation_id: Optional[str] = None

    mode: str
    verb: str

    packs: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)

    final_text: Optional[str] = None
    steps: List[StepExecutionResult] = Field(default_factory=list)

    # cortex-exec currently emits a unix timestamp float
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

    source_service: Optional[str] = None
    source_node: Optional[str] = None

    recall_used: bool = False
    recall_debug: Dict[str, Any] = Field(default_factory=dict)

    metadata: Dict[str, Any] = Field(default_factory=dict)
