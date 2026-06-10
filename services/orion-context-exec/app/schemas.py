from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RepoHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    line_start: int | None = None
    line_end: int | None = None
    snippet: str
    source_ref: str


class RepoFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    truncated: bool = False
    source_ref: str


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    errors: list[str] = Field(default_factory=list)


class SubcallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subcall_id: str
    schema: str | None = None
    ok: bool = True
    result: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class TraceHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handle: str
    source: str
    corr_id: str | None = None
    run_id: str | None = None
    kind: str
    timestamp: str
    snippet: str = ""
    payload_ref: str | None = None


class RecallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[dict[str, Any]] = Field(default_factory=list)
    profile: str = "assist.light.v1"
    query: str = ""
    latency_ms: int = 0
    rendered_head: str = ""
    error: str | None = None
