from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class JournalEntry(BaseModel):
    entry_id: str
    created_at: datetime
    mode: str | None = None
    source_kind: str | None = None
    source_ref: str | None = None
    title: str | None = None
    body: str


class BuildResponse(BaseModel):
    pageindex_impl: str
    pageindex_installation_mode: str
    journal_corpus_row_count: int
    markdown_export_path: str
    pageindex_tree_artifact_path: str
    last_build_started_at: datetime
    last_build_completed_at: datetime
    build_success: bool
    build_error: str | None = None


class StatusResponse(BaseModel):
    pageindex_impl: str
    pageindex_installation_mode: str
    corpus_exists: bool
    journal_corpus_row_count: int
    markdown_export_path: str | None
    pageindex_tree_artifact_path: str | None
    last_build_started_at: datetime | None
    last_build_completed_at: datetime | None
    build_success: bool
    build_error: str | None = None


class QueryRequest(BaseModel):
    query: str
    allow_fallback: bool = False
    top_k: int = 8


class QueryResult(BaseModel):
    node_id: str | None = None
    heading: str | None = None
    excerpt: str
    entry_id: str | None = None
    created_at: str | None = None
    source_kind: str | None = None
    provenance: dict[str, Any] = {}


class QueryResponse(BaseModel):
    pageindex_impl: str
    query_invoked: bool
    query_result_count: int
    fallback_invoked: bool
    results: list[QueryResult]
    metadata: dict[str, Any] = {}
