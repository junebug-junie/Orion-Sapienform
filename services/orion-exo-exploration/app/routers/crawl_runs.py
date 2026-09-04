from __future__ import annotations

from fastapi import APIRouter, Query

from app.storage import repository

router = APIRouter(tags=["crawl-runs"])


def _serialize(row: dict) -> dict:
    out = dict(row)
    for key in ("started_at", "finished_at"):
        if out.get(key) is not None:
            out[key] = out[key].isoformat()
    return out


@router.get("/crawl-runs")
def list_crawl_runs(limit: int = Query(default=50, le=200)):
    rows = repository.list_crawl_runs(limit=limit)
    return {"crawl_runs": [_serialize(r) for r in rows], "count": len(rows)}
