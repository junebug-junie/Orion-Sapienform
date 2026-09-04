from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.storage import repository

router = APIRouter(tags=["finds"])


def _serialize(row: dict) -> dict:
    out = dict(row)
    for key in ("price",):
        if out.get(key) is not None:
            out[key] = float(out[key])
    for key in ("first_seen_at", "last_seen_at", "expires_at", "posted_or_renewed_at"):
        if out.get(key) is not None:
            out[key] = out[key].isoformat()
    return out


@router.get("/finds")
def list_finds(
    category: Optional[str] = Query(default=None),
    min_interest: Optional[float] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=200, le=1000),
):
    rows = repository.list_finds(category=category, min_interest=min_interest, status=status, limit=limit)
    return {"finds": [_serialize(r) for r in rows], "count": len(rows)}


@router.get("/finds/{external_listing_id}")
def get_find(external_listing_id: str):
    row = repository.get_current(external_listing_id)
    if row is None:
        raise HTTPException(status_code=404, detail="find_not_found")
    return _serialize(row)
