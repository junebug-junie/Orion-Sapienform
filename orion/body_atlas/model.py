"""The body atlas artifact: parts -> circulation -> organs, each rung citing the one below.

A file artifact (``python -m orion.body_atlas --out``), not a bus payload. Ids are stable for
the same input, so a later reader (the dig kit, a 3D traversal) can join two atlases by id.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# live: seen on the bus within the live window. dormant: seen, but not lately.
# declared_only: in channels.yaml, never seen (or no live read). transient: seen, below the
# evidence floor and not declared (smokes, probes); listed, kept out of organs.
PartStatus = Literal["live", "dormant", "declared_only", "transient"]
# service: a declared or observed service name. endpoint: a source id with a ':' in it
# (``http:anthropic``, ``operator:...``), i.e. a non-service publisher.
PartKind = Literal["service", "endpoint"]
FlowEvidence = Literal["declared", "observed_publish", "observed_causal"]


class PartV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_id: str
    name: str
    kind: PartKind
    status: PartStatus
    declared: bool
    observed_ids: list[str] = Field(default_factory=list)  # raw bus source ids folded into this part
    publish_count: int = 0
    last_seen_at: str | None = None


class FlowV1(BaseModel):
    """One directed source -> target flow between two parts, summed over channels."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    weight: float
    evidence: list[FlowEvidence]
    channels: list[str] = Field(default_factory=list)
    causal_count: int = 0


class OrganV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organ_id: str
    members: list[str]  # part ids, strongest first
    hub: str
    # The most common channel namespaces (``orion:<ns>:...``) on the organ's internal flows.
    # Code-derived from the data, a hint for naming, not a name.
    namespaces: list[str] = Field(default_factory=list)
    internal_weight: float
    external_weight: float
    name: str | None = None  # written later by Orion; None until then
    named_by: str | None = None


class OrganFlowV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    weight: float


class AtlasSourcesV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_path: str
    catalog_channels: int
    live: bool
    live_graph: str | None = None
    publish_edges: int = 0
    causal_edges: int = 0


class AtlasStatsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts_total: int
    parts_in_organs: int
    transient_parts: int
    flows: int
    organs: int
    singleton_organs: int
    # Newman modularity Q of the organ partition over the undirected part graph. Q near 0 means
    # the "organs" are no better than random groups; ~0.3 and up is real community structure.
    modularity: float
    largest_organ_share: float  # largest organ's members / parts_in_organs


class BodyAtlasV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["body_atlas.v1"] = "body_atlas.v1"
    as_of: str
    sources: AtlasSourcesV1
    parts: list[PartV1]
    flows: list[FlowV1]
    organs: list[OrganV1]
    organ_flows: list[OrganFlowV1]
    stats: AtlasStatsV1
