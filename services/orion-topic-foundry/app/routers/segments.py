from __future__ import annotations

import logging
from uuid import UUID

from typing import Union

from fastapi import APIRouter, HTTPException, Query, Response

from app.models import SegmentFacetsResponse, SegmentFullTextResponse, SegmentListPage, SegmentListResponse, SegmentRawResponse, SegmentRecord
from app.services.data_access import build_full_text, fetch_dataset_rows_by_ids
from app.storage.repository import count_segments, fetch_dataset, fetch_run, fetch_segment, fetch_segments, segment_facets


logger = logging.getLogger("topic-foundry.segments")

router = APIRouter()


@router.get("/segments", response_model=Union[SegmentListResponse, SegmentListPage])
def list_segments(
    run_id: UUID,
    response: Response,
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
def get_segment(segment_id: UUID, include_full_text: bool = Query(default=False)) -> SegmentRecord:
    row = fetch_segment(segment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Segment not found")
    full_text = None
    if include_full_text:
        run = fetch_run(UUID(row["run_id"]))
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        dataset = fetch_dataset(UUID(run["dataset_id"]))
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        provenance = row.get("provenance") or {}
        row_ids = provenance.get("row_ids") or []
        if isinstance(row_ids, str):
            row_ids = [row_ids]
        rows = fetch_dataset_rows_by_ids(dataset=dataset, row_ids=row_ids)
        full_text = build_full_text(rows, dataset.text_columns) if rows else (row.get("snippet") or "")
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
        full_text=full_text,
    )


@router.get("/segments/{segment_id}/raw", response_model=SegmentRawResponse)
def get_segment_raw(segment_id: UUID) -> SegmentRawResponse:
    row = fetch_segment(segment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Segment not found")
    return SegmentRawResponse(segment_id=segment_id, provenance=row["provenance"])


@router.get("/segments/{segment_id}/full_text", response_model=SegmentFullTextResponse)
def get_segment_full_text(segment_id: UUID) -> SegmentFullTextResponse:
    row = fetch_segment(segment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Segment not found")
    run = fetch_run(UUID(row["run_id"]))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    dataset = fetch_dataset(UUID(run["dataset_id"]))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    provenance = row.get("provenance") or {}
    row_ids = provenance.get("row_ids") or []
    if isinstance(row_ids, str):
        row_ids = [row_ids]
    rows = fetch_dataset_rows_by_ids(dataset=dataset, row_ids=row_ids)
    full_text = build_full_text(rows, dataset.text_columns) if rows else (row.get("snippet") or "")
    return SegmentFullTextResponse(
        segment_id=segment_id,
        run_id=UUID(row["run_id"]),
        full_text=full_text,
        chars=len(full_text),
        row_ids_count=len(row_ids),
    )
