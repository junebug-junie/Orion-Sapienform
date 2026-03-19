from __future__ import annotations

import logging
from uuid import UUID

from typing import Union

from fastapi import APIRouter, HTTPException, Query, Response

from app.models import SegmentFacetsResponse, SegmentListPage, SegmentListResponse, SegmentRawResponse, SegmentRecord
from app.storage.repository import count_segments, fetch_segment, fetch_segments, segment_facets


logger = logging.getLogger("topic-foundry.segments")

router = APIRouter()


@router.get("/segments", response_model=Union[SegmentListResponse, SegmentListPage])
def list_segments(
    run_id: UUID,
    q: str | None = Query(default=None),
    aspect: str | None = Query(default=None),
    has_enrichment: bool | None = Query(default=None),
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc"),
    include_snippet: bool = Query(default=False),
    include_bounds: bool = Query(default=False),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    format: str | None = Query(default=None),
    response: Response,
):
    rows = fetch_segments(
        run_id,
        q=q,
        aspect=aspect,
        has_enrichment=has_enrichment,
        sort_by=sort_by,
        sort_dir=sort_dir,
        limit=limit,
        offset=offset,
    )
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
            start_at=row.get("start_at") if include_bounds else None,
            end_at=row.get("end_at") if include_bounds else None,
        )
        for row in rows
    ]
    total = count_segments(run_id, aspect=aspect, has_enrichment=has_enrichment, q=q)
    response.headers["X-Limit"] = str(limit)
    response.headers["X-Offset"] = str(offset)
    response.headers["X-Total-Count"] = str(total)
    if format == "wrapped":
        return SegmentListPage(run_id=run_id, items=segments, limit=limit, offset=offset, total=total)
    return SegmentListResponse(run_id=run_id, segments=segments)


@router.get("/segments/facets", response_model=SegmentFacetsResponse)
def get_segment_facets(
    run_id: UUID,
    q: str | None = Query(default=None),
    aspect: str | None = Query(default=None),
    has_enrichment: bool | None = Query(default=None),
):
    return segment_facets(run_id, q=q, aspect=aspect, has_enrichment=has_enrichment)


@router.get("/segments/{segment_id}", response_model=SegmentRecord)
def get_segment(segment_id: UUID) -> SegmentRecord:
    row = fetch_segment(segment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Segment not found")
    return SegmentRecord(
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
        snippet=row.get("snippet"),
        chars=row.get("chars"),
        row_ids_count=row.get("row_ids_count"),
        start_at=row.get("start_at"),
        end_at=row.get("end_at"),
    )


@router.get("/segments/{segment_id}/raw", response_model=SegmentRawResponse)
def get_segment_raw(segment_id: UUID) -> SegmentRawResponse:
    row = fetch_segment(segment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Segment not found")
    return SegmentRawResponse(segment_id=segment_id, provenance=row["provenance"])
