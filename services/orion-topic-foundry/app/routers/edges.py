from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Query

from app.models import KgEdgeListPage, KgEdgeListResponse, KgEdgeRecord
from app.storage.repository import list_edges, list_edges_filtered


logger = logging.getLogger("topic-foundry.edges")

router = APIRouter()


@router.get("/edges", response_model=KgEdgeListResponse)
def list_edges_endpoint(run_id: UUID, limit: int = Query(default=200, ge=1, le=500)) -> KgEdgeListResponse:
    rows = list_edges(run_id, limit=limit)
    edges = [
        KgEdgeRecord(
            edge_id=UUID(row["edge_id"]),
            segment_id=UUID(row["segment_id"]),
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row["confidence"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
    return KgEdgeListResponse(run_id=run_id, edges=edges)


@router.get("/kg/edges", response_model=KgEdgeListPage)
def list_kg_edges_endpoint(
    run_id: UUID,
    q: str | None = None,
    predicate: str | None = None,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> KgEdgeListPage:
    rows = list_edges_filtered(run_id, q=q, predicate=predicate, limit=limit, offset=offset)
    edges = [
        KgEdgeRecord(
            edge_id=UUID(row["edge_id"]),
            segment_id=UUID(row["segment_id"]),
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row["confidence"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
    return KgEdgeListPage(run_id=run_id, items=edges, limit=limit, offset=offset)
