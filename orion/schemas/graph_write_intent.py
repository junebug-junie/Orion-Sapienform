"""Graph write intent contract for persistence routing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

GRAPH_WRITE_INTENT_KIND = "graph.write_intent.v1"

GraphWriteOperationV1 = Literal[
    "upsert_node",
    "upsert_edge",
    "delete_node",
    "delete_edge",
    "append_event",
]


class GraphWriteNodePayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    id: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphWriteEdgePayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicate: str
    source_id: str
    target_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphWriteProvenanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    producer: str
    source_refs: List[str] = Field(default_factory=list, max_length=32)
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)


class GraphWriteCompatibilityV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rdf_graph_name: Optional[str] = None


class GraphWriteIntentV1(BaseModel):
    """In-process write intent for graph persistence routing."""

    model_config = ConfigDict(extra="forbid")

    workload: str
    operation: GraphWriteOperationV1
    identity_key: str = Field(min_length=1)
    node: Optional[GraphWriteNodePayloadV1] = None
    edge: Optional[GraphWriteEdgePayloadV1] = None
    provenance: GraphWriteProvenanceV1
    compatibility: Optional[GraphWriteCompatibilityV1] = None
    routing_hint: Optional[str] = None
