from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class TopicSummaryEventV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_version: str
    window_start: datetime
    window_end: datetime
    topic_id: int
    topic_label: Optional[str] = None
    topic_keywords: List[str] = []
    doc_count: int
    pct_of_window: float


class TopicShiftEventV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_version: str
    window_start: datetime
    window_end: datetime
    session_id: str
    turns: int
    unique_topics: int
    entropy: float
    switch_rate: float
    dominant_topic_id: Optional[int] = None
    dominant_pct: Optional[float] = None


class TopicRailAssignedV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_version: str
    node_name: str
    doc_count: int
    outlier_pct: float
    top_topic_ids: List[int]
    created_at: datetime
