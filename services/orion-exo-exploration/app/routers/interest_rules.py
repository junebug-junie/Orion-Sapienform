from __future__ import annotations

from fastapi import APIRouter

from app.storage import repository

router = APIRouter(tags=["interest-rules"])


def _serialize(row: dict) -> dict:
    out = dict(row)
    if out.get("created_at") is not None:
        out["created_at"] = out["created_at"].isoformat()
    for key in ("min_price", "max_price"):
        if out.get(key) is not None:
            out[key] = float(out[key])
    return out


@router.get("/interest-rules")
def list_interest_rules():
    rows = repository.list_interest_rules()
    return {"interest_rules": [_serialize(r) for r in rows], "count": len(rows)}
