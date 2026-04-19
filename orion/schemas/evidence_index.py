from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceUnitV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: str
    unit_kind: str
    source_family: str
    source_kind: str
    source_ref: str
    correlation_id: str | None = None
    parent_unit_id: str | None = None
    sibling_prev_id: str | None = None
    sibling_next_id: str | None = None
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    facets: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime | None = None


class EvidenceQueryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_query: str | None = None
    unit_kinds: list[str] = Field(default_factory=list)
    source_family: list[str] = Field(default_factory=list)
    source_kind: list[str] = Field(default_factory=list)
    source_ref: str | None = None
    correlation_id: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    required_facets: list[str] = Field(default_factory=list)
    search_level: Literal["document", "section", "leaf", "auto"] = "auto"
    include_parent_context: bool = False
    include_child_context: bool = False
    limit: int = 50
    offset: int = 0


class EvidenceQueryResultItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: str
    unit_kind: str
    source_kind: str
    source_ref: str
    title: str | None = None
    summary: str | None = None
    created_at: datetime
    matched_facets: list[str] = Field(default_factory=list)
    matched_fields: list[str] = Field(default_factory=list)
    applied_filters: list[str] = Field(default_factory=list)
    provenance: dict[str, str | None] = Field(default_factory=dict)
    parent_context: dict[str, Any] | None = None
    child_context: list[dict[str, Any]] = Field(default_factory=list)


class MarkdownSpecIngestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    body: str
    source_ref: str
    source_kind: str = "markdown_spec"
    correlation_id: str | None = None
    created_at: datetime
