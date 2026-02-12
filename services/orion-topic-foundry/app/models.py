from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DatasetSpec(BaseModel):
    dataset_id: UUID
    name: str
    source_table: str
    id_column: str
    time_column: str
    text_columns: List[str]
    timezone: str = "UTC"
    boundary_column: Optional[str] = None
    boundary_strategy: Optional[Literal["column"]] = None
    created_at: datetime


class WindowingSpec(BaseModel):
    windowing_mode: Literal["document", "time_gap", "conversation_bound"] = "document"
    time_gap_minutes: int = 15
    max_chars: int = 6000
    conversation_bound: Optional[str] = None
    boundary_column: Optional[str] = None


class ModelSpec(BaseModel):
    algorithm: Literal["hdbscan"] = "hdbscan"
    embedding_source_url: Optional[str] = None
    min_cluster_size: int = 15
    metric: str = "cosine"
    params: Dict[str, Any] = Field(default_factory=dict)
    model_meta: Dict[str, Any] = Field(default_factory=dict)


class EnrichmentSpec(BaseModel):
    enable_enrichment: bool = False
    enricher: Literal["llm", "heuristic"] = "llm"
    aspect_taxonomy: Optional[str] = None


class RunSpecSnapshot(BaseModel):
    dataset: DatasetSpec
    windowing: WindowingSpec
    model: ModelSpec
    enrichment: EnrichmentSpec = Field(default_factory=EnrichmentSpec)
    run_scope: Optional[Literal["micro", "macro"]] = None


class RunRecord(BaseModel):
    run_id: UUID
    model_id: UUID
    dataset_id: UUID
    specs: RunSpecSnapshot
    spec_hash: Optional[str] = None
    status: Literal["queued", "running", "complete", "failed"]
    stage: Optional[str] = None
    run_scope: Optional[Literal["micro", "macro"]] = None
    stats: Dict[str, Any] = Field(default_factory=dict)
    artifact_paths: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class SegmentRecord(BaseModel):
    segment_id: UUID
    run_id: UUID
    size: int
    provenance: Dict[str, Any]
    created_at: datetime
    label: Optional[str] = None
    topic_id: Optional[int] = None
    topic_prob: Optional[float] = None
    is_outlier: Optional[bool] = None
    title: Optional[str] = None
    aspects: Optional[List[str]] = None
    sentiment: Optional[Dict[str, Any]] = None
    meaning: Optional[Dict[str, Any]] = None
    enrichment: Optional[Dict[str, Any]] = None
    enriched_at: Optional[datetime] = None
    enrichment_version: Optional[str] = None
    snippet: Optional[str] = None
    chars: Optional[int] = None
    row_ids_count: Optional[int] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    full_text: Optional[str] = None


class DatasetCreateRequest(BaseModel):
    name: str
    source_table: str
    id_column: str
    time_column: str
    text_columns: List[str]
    timezone: Optional[str] = None
    boundary_column: Optional[str] = None
    boundary_strategy: Optional[Literal["column"]] = None


class DatasetUpdateRequest(BaseModel):
    name: Optional[str] = None
    source_table: Optional[str] = None
    id_column: Optional[str] = None
    time_column: Optional[str] = None
    text_columns: Optional[List[str]] = None
    timezone: Optional[str] = None
    boundary_column: Optional[str] = None
    boundary_strategy: Optional[Literal["column"]] = None


class DatasetCreateResponse(BaseModel):
    dataset_id: UUID
    created_at: datetime


class DatasetPreviewRequest(BaseModel):
    dataset_id: Optional[UUID] = None
    dataset: Optional[DatasetSpec] = None
    windowing: Optional[WindowingSpec] = None
    windowing_spec: Optional[WindowingSpec] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    limit: int = 200


class DatasetPreviewDoc(BaseModel):
    doc_id: str
    segment_id: str
    row_ids_count: int
    chars: int
    snippet: str


class DatasetPreviewResponse(BaseModel):
    rows_scanned: int
    row_count: int
    blocks_generated: int
    segments_generated: int
    segment_count: int
    docs_generated: int
    doc_count: int
    avg_chars: float
    p95_chars: int
    max_chars: int
    observed_start_at: Optional[datetime] = None
    observed_end_at: Optional[datetime] = None
    samples: List[DatasetPreviewDoc]


class DatasetPreviewDetailResponse(BaseModel):
    dataset_id: UUID
    doc_id: str
    full_text: str
    char_count: int
    observed_range: Dict[str, Optional[str]] = Field(default_factory=dict)
    is_truncated: bool = False


class ModelCreateRequest(BaseModel):
    name: str
    version: str
    stage: Optional[str] = "development"
    dataset_id: UUID
    model_spec: ModelSpec
    windowing_spec: WindowingSpec
    enrichment_spec: Optional[EnrichmentSpec] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_meta: Dict[str, Any] = Field(default_factory=dict)


class ModelCreateResponse(BaseModel):
    model_id: UUID
    created_at: datetime


class ModelSummary(BaseModel):
    model_id: UUID
    name: str
    version: str
    stage: Optional[str]
    dataset_id: UUID
    model_meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ModelListResponse(BaseModel):
    models: List[ModelSummary]


class ModelVersionEntry(BaseModel):
    model_id: UUID
    name: str
    version: str
    stage: Optional[str]
    created_at: datetime


class ModelVersionsResponse(BaseModel):
    name: str
    versions: List[ModelVersionEntry]


class DatasetListResponse(BaseModel):
    datasets: List[DatasetSpec]


class RunSummary(BaseModel):
    run_id: UUID
    model_id: UUID
    dataset_id: UUID
    status: Literal["queued", "running", "complete", "failed"]
    stage: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class RunListResponse(BaseModel):
    runs: List[RunSummary]


class RunTrainRequest(BaseModel):
    model_id: UUID
    dataset_id: UUID
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    run_scope: Optional[Literal["micro", "macro"]] = None
    windowing_spec: Optional[WindowingSpec] = None
    topic_mode: Literal["standard", "guided", "zeroshot", "dynamic", "class_based", "long_document", "hierarchical"] = "standard"
    topic_mode_params: Dict[str, Any] = Field(default_factory=dict)


class RunEnrichRequest(BaseModel):
    limit: Optional[int] = None
    force: bool = False
    enricher: Optional[Literal["llm", "heuristic"]] = None
    target: Literal["segments", "topics", "both"] = "segments"
    fields: List[Literal["title", "aspects", "meaning", "sentiment"]] = Field(default_factory=list)
    llm_backend: Optional[str] = None
    prompt_template: Optional[str] = None


class RunEnrichResponse(BaseModel):
    run_id: UUID
    status: Literal["queued", "running", "complete", "failed"]
    enriched_count: int
    failed_count: int


class RunTrainResponse(BaseModel):
    run_id: UUID
    status: Literal["queued", "running", "complete", "failed"]
    doc_count: Optional[int] = None
    cluster_count: Optional[int] = None
    outlier_rate: Optional[float] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    model_meta_used: Dict[str, Any] = Field(default_factory=dict)
    topic_mode: Optional[str] = None
    topic_mode_params: Dict[str, Any] = Field(default_factory=dict)


class SegmentListResponse(BaseModel):
    run_id: UUID
    segments: List[SegmentRecord]


class SegmentListPage(BaseModel):
    run_id: UUID
    items: List[SegmentRecord]
    limit: int
    offset: int
    total: Optional[int] = None


class SegmentFacetCount(BaseModel):
    key: str
    count: int


class SegmentFacetsTotals(BaseModel):
    segments: int
    enriched: int


class SegmentFacetsResponse(BaseModel):
    aspects: List[SegmentFacetCount]
    intents: List[SegmentFacetCount]
    friction_buckets: List[SegmentFacetCount]
    totals: SegmentFacetsTotals


class TopicSummaryItem(BaseModel):
    topic_id: int
    count: int
    outlier_pct: Optional[float] = None
    label: Optional[str] = None
    scope: Optional[Literal["micro", "macro"]] = None
    parent_topic_id: Optional[int] = None


class TopicSummaryPage(BaseModel):
    items: List[TopicSummaryItem]
    limit: int
    offset: int
    total: Optional[int] = None


class TopicKeywordsResponse(BaseModel):
    topic_id: int
    keywords: List[str]


class RunCompareResponse(BaseModel):
    left_run_id: UUID
    right_run_id: UUID
    left_stats: Dict[str, Any]
    right_stats: Dict[str, Any]
    diffs: Dict[str, Any]
    aspect_diffs: List[Dict[str, Any]]


class SegmentRawResponse(BaseModel):
    segment_id: UUID
    provenance: Dict[str, Any]


class SegmentFullTextResponse(BaseModel):
    segment_id: UUID
    run_id: UUID
    full_text: str
    chars: int
    row_ids_count: int


class CapabilitiesResponse(BaseModel):
    capabilities: Dict[str, Any]
    backends: Dict[str, List[str]]


class DriftRunRequest(BaseModel):
    model_name: str
    window_days: Optional[int] = None
    window_hours: Optional[int] = None
    threshold_js: Optional[float] = None
    threshold_outlier: Optional[float] = None


class DriftRunResponse(BaseModel):
    drift_id: UUID
    status: Literal["complete", "skipped", "error"]


class RunListItem(BaseModel):
    run_id: UUID
    status: Literal["queued", "running", "complete", "failed"]
    stage: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model: Dict[str, Any]
    dataset: Dict[str, Any]
    window: Dict[str, Any]
    stats_summary: Dict[str, Any]


class RunListPage(BaseModel):
    items: List[RunListItem]
    limit: int
    offset: int
    total: Optional[int] = None


class DriftRecord(BaseModel):
    drift_id: UUID
    model_id: UUID
    window_start: datetime
    window_end: datetime
    js_divergence: float
    outlier_pct: float
    threshold_js: Optional[float] = None
    threshold_outlier: Optional[float] = None
    outlier_pct_delta: Optional[float] = None
    top_topic_share_delta: Optional[float] = None
    topic_shares: Dict[str, Any]
    created_at: datetime


class DriftListResponse(BaseModel):
    model_name: str
    records: List[DriftRecord]


class KgEdgeRecord(BaseModel):
    edge_id: UUID
    segment_id: UUID
    subject: str
    predicate: str
    object: str
    confidence: float
    created_at: datetime


class KgEdgeListResponse(BaseModel):
    run_id: UUID
    edges: List[KgEdgeRecord]


class KgEdgeListPage(BaseModel):
    run_id: UUID
    items: List[KgEdgeRecord]
    limit: int
    offset: int


class EventRecord(BaseModel):
    event_id: UUID
    kind: str
    run_id: Optional[UUID] = None
    model_id: Optional[UUID] = None
    drift_id: Optional[UUID] = None
    payload: Optional[Dict[str, Any]] = None
    bus_status: Optional[str] = None
    bus_error: Optional[str] = None
    created_at: datetime


class EventListResponse(BaseModel):
    items: List[EventRecord]
    limit: int
    offset: int
