"""Bus payload schema for the codebase-mass domain's ``orion:substrate:codebase_delta``
channel (docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md). Each
event carries exactly one producer domain's real delta -- git, PR lifecycle, or
graphify structural -- since each producer runs on its own independent interval
(the design spec's "Dedicated service" section); the other two nested fields are
``None`` on any given event, not a fabricated zero.

Field shapes mirror ``orion/structural_mass/{git_delta,pr_lifecycle,graph_delta}.py``'s
own dataclasses as closely as pydantic/JSON allows (e.g. tuples become lists, since
JSON has no tuple type) -- this schema is the wire format those pure dataclasses get
serialized into, not an independent redesign of their shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GitDeltaPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prev_sha: str
    head_sha: str
    commit_count: int
    files_added: int
    files_deleted: int
    files_modified: int
    lines_added: int
    lines_removed: int


class PrLifecycleDeltaPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    since: datetime
    until: datetime
    submitted_count: int
    merged_count: int
    closed_without_merge_count: int
    submitted_numbers: list[int] = Field(default_factory=list)
    merged_numbers: list[int] = Field(default_factory=list)
    closed_without_merge_numbers: list[int] = Field(default_factory=list)
    possibly_truncated: bool = False


class GraphDeltaPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_count_delta: int
    edge_count_delta: int
    community_count_delta: int
    # None (not a fabricated 0.0/1.0) when the source snapshots' god_nodes could
    # not be determined -- see orion/structural_mass/graph_delta.py's docstring.
    god_node_jaccard_similarity: float | None = None
    god_nodes_entered: list[str] | None = None
    god_nodes_exited: list[str] | None = None


class CodebaseDeltaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["codebase_delta.v1"] = "codebase_delta.v1"
    # Which one of git/pr_lifecycle/graph fired this event -- the other two
    # nested fields are None on any given event, never a fabricated zero delta
    # for a producer that simply didn't run this tick.
    domain: Literal["git", "pr_lifecycle", "graph"]
    observed_at: datetime
    git: GitDeltaPayloadV1 | None = None
    pr_lifecycle: PrLifecycleDeltaPayloadV1 | None = None
    graph: GraphDeltaPayloadV1 | None = None

    @model_validator(mode="after")
    def _exactly_one_domain_populated(self) -> "CodebaseDeltaV1":
        """Enforces the invariant this schema's own docstring only states in
        prose (flagged by code review 2026-07-30): the field matching
        ``domain`` must be populated, and the other two must be ``None`` --
        not just as a convention producers are expected to follow, but as
        something a bad payload actually fails validation on. Without this,
        a malformed event (e.g. ``domain="git"`` with ``pr_lifecycle`` also
        set, or ``domain="git"`` with ``git=None``) would pass schema
        validation silently and only misbehave once a real consumer reads
        the wrong/missing field -- exactly the kind of "schema-valid payload
        with meaningless content" root CLAUDE.md's "no empty-shell
        cognition" section warns about."""
        fields = {"git": self.git, "pr_lifecycle": self.pr_lifecycle, "graph": self.graph}
        populated = [name for name, value in fields.items() if value is not None]
        if populated != [self.domain]:
            raise ValueError(
                f"domain={self.domain!r} requires exactly the {self.domain!r} field "
                f"populated and the other two None; got populated={populated!r}"
            )
        return self
