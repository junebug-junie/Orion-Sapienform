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
    # Wall-clock timestamps parallel to recent_perturbations (same length,
    # same index alignment -- recent_perturbation_at[i] is when
    # recent_perturbations[i] was first seen). Added 2026-07-16 alongside the
    # saturating-counter fix: services/orion-field-digester/app/digestion/
    # perturbation.py now prunes recent_perturbations by wall-clock recency
    # instead of a fixed last-20-labels-ever cap (the old cap saturated
    # self_state's recent_perturbation_count to 1.0 within the first few
    # ticks of any real corpus and then pinned it there permanently -- see
    # perturbation.py's module docstring for the live evidence). Both lists
    # are always written/pruned together; a FieldStateV1 persisted before
    # this fix will load with an empty recent_perturbation_at, so the length
    # mismatch resets recent_perturbations to empty on the very first tick
    # after upgrade (zip() in perturbation.py's pruning stops at the shorter
    # list) -- an immediate, one-tick reset, not gated by window expiry --
    # and it repopulates normally from there.
    recent_perturbation_at: list[datetime] = Field(default_factory=list)
    topology_id: str | None = None
    topology_version: str | None = None
    topology_loaded_from: str | None = None
