from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FieldEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    edge_type: Literal["node_capability", "node_service", "service_organ", "capability_cognitive", "node_dependency", "capability_capability"]
    weight: float = Field(ge=0.0, le=1.0)
    channel_map: dict[str, str] = Field(default_factory=dict)
    weight_source: Literal["designed", "learned"] = "designed"
    learned_at: datetime | None = None


class FieldStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["field.state.v1"] = "field.state.v1"
    generated_at: datetime
    tick_id: str
    node_vectors: dict[str, dict[str, float]] = Field(default_factory=dict)
    capability_vectors: dict[str, dict[str, float]] = Field(default_factory=dict)
    # Phase 3 (2026-07-12, self-state/mesh substrate redesign): keyed by
    # capability target_id -> channel -> the edge source_id (a node_id like
    # "node:atlas" or another capability_id for a capability_capability edge)
    # that contributed the single largest weighted amount to that channel in
    # the most recent diffusion pass (services/orion-field-digester/app/
    # digestion/diffusion.py). A simple "primary contributor this tick"
    # proxy, not a full historical attribution ledger -- capability_vectors
    # values accumulate across ticks, but tracking exact per-tick provenance
    # of an accumulated value would be a bigger feature than this phase
    # needs. Lets self-state's evidence say which node is behind a pressure
    # instead of an anonymous capability-level number.
    capability_provenance: dict[str, dict[str, str]] = Field(default_factory=dict)
    edges: list[FieldEdgeV1] = Field(default_factory=list)
    recent_perturbations: list[str] = Field(default_factory=list)
    topology_id: str | None = None
    topology_version: str | None = None
    topology_loaded_from: str | None = None
