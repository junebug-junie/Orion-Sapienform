from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models import SegmentListPage, SegmentRecord, TopicKeywordsResponse, TopicSummaryItem, TopicSummaryPage
from app.settings import settings
from app.storage.repository import fetch_run, fetch_topic_segments, list_topics


logger = logging.getLogger("topic-foundry.topics")

router = APIRouter()


@router.get("/topics", response_model=TopicSummaryPage)
def list_topics_endpoint(
    run_id: UUID,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    rows, total = list_topics(run_id, limit=limit, offset=offset)
    items = []
    for row in rows:
        count = int(row.get("count") or 0)
        outliers = int(row.get("outliers") or 0)
        outlier_pct = float(outliers) / float(count) if count and outliers else None
        topic_id = row.get("topic_id")
        items.append(
            TopicSummaryItem(
                topic_id=int(topic_id) if topic_id is not None else -1,
                count=count,
                outlier_pct=outlier_pct,
                label=None,
                scope=row.get("scope"),
                parent_topic_id=row.get("parent_topic_id"),
            )
        )
    return TopicSummaryPage(items=items, limit=limit, offset=offset, total=total)


@router.get("/topics/{topic_id}/segments", response_model=SegmentListPage)
def list_topic_segments_endpoint(
    topic_id: int,
    run_id: UUID,
    include_snippet: bool = Query(default=True),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    rows = fetch_topic_segments(run_id, topic_id, limit=limit, offset=offset)
    segments = [
        SegmentRecord(
            segment_id=UUID(row["segment_id"]),
            run_id=UUID(row["run_id"]),
            size=row["size"],
            provenance=row["provenance"],
            label=row.get("label"),
            created_at=row["created_at"],
            topic_id=row.get("topic_id"),
            topic_prob=row.get("topic_prob"),
            is_outlier=row.get("is_outlier"),
            title=row.get("title"),
            aspects=row.get("aspects"),
            sentiment=row.get("sentiment"),
            meaning=row.get("meaning"),
            enrichment=row.get("enrichment"),
            enriched_at=row.get("enriched_at"),
            enrichment_version=row.get("enrichment_version"),
            snippet=row.get("snippet") if include_snippet else None,
            chars=row.get("chars") if include_snippet else None,
            row_ids_count=row.get("row_ids_count") if include_snippet else None,
            start_at=row.get("start_at"),
            end_at=row.get("end_at"),
        )
        for row in rows
    ]
    return SegmentListPage(run_id=run_id, items=segments, limit=limit, offset=offset, total=None)


@router.get("/topics/{topic_id}/keywords", response_model=TopicKeywordsResponse)
def topic_keywords_endpoint(topic_id: int, run_id: UUID):
    run_row = fetch_run(run_id)
    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")
    artifact_paths = run_row.get("artifact_paths") or {}
    keywords_path = artifact_paths.get("topics_keywords")
    if keywords_path and Path(keywords_path).exists():
        data = json.loads(Path(keywords_path).read_text())
        keywords = data.get(str(topic_id), [])
        return TopicKeywordsResponse(topic_id=topic_id, keywords=keywords)
    return TopicKeywordsResponse(topic_id=topic_id, keywords=[])
