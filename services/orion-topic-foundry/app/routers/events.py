from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Query

from app.models import EventListResponse, EventRecord
from app.storage.repository import list_events


logger = logging.getLogger("topic-foundry.events")

router = APIRouter()


@router.get("/events", response_model=EventListResponse)
def list_events_endpoint(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    kind: str | None = None,
) -> EventListResponse:
    rows = list_events(limit=limit, offset=offset, kind=kind)
    items = [
        EventRecord(
            event_id=UUID(row["event_id"]),
            kind=row["kind"],
            run_id=UUID(row["run_id"]) if row.get("run_id") else None,
            model_id=UUID(row["model_id"]) if row.get("model_id") else None,
            drift_id=UUID(row["drift_id"]) if row.get("drift_id") else None,
            payload=row.get("payload"),
            bus_status=row.get("bus_status"),
            bus_error=row.get("bus_error"),
            created_at=row["created_at"],
        )
        for row in rows
    ]
    return EventListResponse(items=items, limit=limit, offset=offset)
