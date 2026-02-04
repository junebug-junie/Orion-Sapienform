from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TopicFoundryRunCompleteV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: UUID
    model_id: UUID
    dataset_id: UUID
    model_name: str
    model_version: str
    status: str
    stats: dict
    completed_at: Optional[datetime] = None


class TopicFoundryEnrichCompleteV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: UUID
    model_id: UUID
    dataset_id: UUID
    model_name: str
    model_version: str
    status: str
    enriched_count: int
    failed_count: int
    completed_at: Optional[datetime] = None


class TopicFoundryDriftAlertV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    drift_id: UUID
    model_id: UUID
    model_name: str
    window_start: datetime
    window_end: datetime
    js_divergence: float
    outlier_pct_delta: float
    top_topic_share_delta: float
    threshold_js: Optional[float] = None
    threshold_outlier: Optional[float] = None
    created_at: datetime


class KgEdgeIngestItemV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    edge_id: UUID
    segment_id: UUID
    subject: str
    predicate: str
    object: str
    confidence: float
    created_at: datetime


class KgEdgeIngestV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: UUID
    model_id: UUID
    model_name: str
    edges: List[KgEdgeIngestItemV1]
