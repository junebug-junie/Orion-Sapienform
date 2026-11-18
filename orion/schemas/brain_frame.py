from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SUBSTRATE_BRAIN_FRAME_KIND = "substrate.brain_frame.v1"


class BrainRegionV1(BaseModel):
    """The spine of the contract: a continuous, trackable per-region signal."""

    model_config = ConfigDict(extra="forbid")

    # "self_state" removed 2026-10-07: its only source table
    # (substrate_self_state) has had 0 rows since the 2026-07-22 SelfStateV1
    # burn, so no frame has carried a self_state region since (0 of 4,743
    # frames in substrate_brain_frame_log, checked live 2026-10-07).
    dimension: Literal[
        "node_kind", "lane", "lattice_layer", "honesty_metrics", "field_anomaly"
    ]
    region_id: str
    label: str
    intensity: float = Field(ge=0.0, le=1.0)
    state: Literal["firing", "steady", "starving"]
    node_count: int = Field(default=0, ge=0)
    as_of: datetime
    stale: bool = False
    detail: dict[str, float] = Field(default_factory=dict)


class BrainSpotlightV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attended_node_ids: list[str] = Field(default_factory=list)
    dwell_ticks: int = Field(default=0, ge=0)
    coalition_stability: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str | None = None
    as_of: datetime
    stale: bool = False


class BrainNodeSampleV1(BaseModel):
    """Best-effort decoration. NO continuity guarantee across frames."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_kind: str
    activation: float = Field(ge=0.0, le=1.0)
    pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    dormant: bool = False
    label: str = ""


class BrainEdgeSampleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str
    weight: float = Field(default=0.0, ge=0.0, le=1.0)


class SubstrateBrainFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.brain_frame.v1"] = "substrate.brain_frame.v1"

    frame_id: str
    generated_at: datetime
    tick_seq: int = Field(ge=0)
    phase: Literal["warming", "live"] = "warming"
    source: str = "orion-substrate-runtime"

    regions: list[BrainRegionV1] = Field(default_factory=list)
    spotlight: BrainSpotlightV1 | None = None
    nodes: list[BrainNodeSampleV1] = Field(default_factory=list)
    edges: list[BrainEdgeSampleV1] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
