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
# host: a physical machine (``host:athena``); never an organ member (placement is not function).
# sensor: a camera stream (``camera:walkway``). store: a Postgres table, Chroma collection or
# FalkorDB graph (``pg:``, ``chroma:``, ``falkor:``). process: a durable workflow or a substrate
# reducer (``run:``, ``reducer:``). See sources.py for where each comes from.
PartKind = Literal["service", "endpoint", "host", "sensor", "store", "process"]
FlowEvidence = Literal[
    "declared", "observed_publish", "observed_causal",
    # runs_on: a container seen running on that host's docker / a host allowlist file says so
    "observed_placement", "declared_placement",
    # sees_through: a service's config names the camera stream (frame router policy, *STREAM_ID env)
    "declared_config",
    # writes: sql-writer route map (message kind -> table) joined with that kind's channel producers
    "declared_route",
    # uses: a service's .env_example names the store (read or write: the env does not say which)
    "declared_env",
    # runs: the producer of the run-state channel (channels.yaml) / no machine-readable runner
    # exists, so the service that owns this family's state table (labeled as an assumption)
    "declared_runner", "assumed_runner",
]
# signals: bus circulation (the v1 relation). runs_on: service -> host. sees_through: camera ->
# service. writes: producer -> store. uses: service -> store. runs: runner -> process.
FlowRelation = Literal["signals", "runs_on", "sees_through", "writes", "uses", "runs"]


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
    # small, source-derived facts for reading (hardware, role, rows, stream, last_run, ...)
    facts: dict[str, str] = Field(default_factory=dict)


class FlowV1(BaseModel):
    """One directed source -> target flow between two parts, summed over channels."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    weight: float
    evidence: list[FlowEvidence]
    channels: list[str] = Field(default_factory=list)
    causal_count: int = 0
    relation: FlowRelation = "signals"


class OrganV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organ_id: str
    members: list[str]  # part ids, strongest first
    hub: str
    kinds: dict[str, int] = Field(default_factory=dict)  # member count by part kind
    placement: dict[str, int] = Field(default_factory=dict)  # host name -> members running there
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


class SourceReadV1(BaseModel):
    """One source the atlas tried to read. ``unread`` is a failure (error in ``detail``) and is
    printed with every atlas; ``absent`` is a declared file this checkout does not have."""

    model_config = ConfigDict(extra="forbid")

    name: str
    origin: Literal["declared", "live"]
    status: Literal["read", "unread", "absent"]
    detail: str = ""


class AtlasSourcesV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_path: str
    catalog_channels: int
    live: bool
    live_graph: str | None = None
    publish_edges: int = 0
    causal_edges: int = 0
    reads: list[SourceReadV1] = Field(default_factory=list)


class AtlasStatsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts_total: int  # hosts included
    parts_in_organs: int
    hosts: int = 0  # host parts: listed, never in organs
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
    # what the atlas cannot see yet (no machine-readable source); printed, never faked
    known_gaps: list[str] = Field(default_factory=list)
    # raw names (containers, config service names) that matched no part, by evidence label
    unmatched: dict[str, list[str]] = Field(default_factory=dict)
    # stores a service's env expects that do not exist live (reported, never parts)
    missing_declared_stores: list[str] = Field(default_factory=list)
