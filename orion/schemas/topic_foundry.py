from __future__ import annotations

from datetime import datetime
from typing import Optional
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


# KgEdgeIngestItemV1 / KgEdgeIngestV1 (bus envelope for the retired
# orion:kg:edge:ingest.v1 channel) removed 2026-07-28 -- zero live consumers
# ever subscribed. The same underlying data (topic-foundry's typed
# mention/asks_about/claims_about/next_step edges) now reaches a real
# consumer via GET /kg/edges (KgEdgeRecord in
# services/orion-topic-foundry/app/models.py), pulled by
# orion-hub/scripts/concept_atlas_routes.py into the live Falkor substrate
# graph. See orion/substrate/adapters/topic_foundry.py's module docstring.
