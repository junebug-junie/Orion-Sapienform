# ==================================================
# dream_api.py
# ==================================================
from fastapi import APIRouter, HTTPException, Query
from datetime import date
from .wake_readout import build_readout

router = APIRouter()

@router.get("/wakeup/today")
def wake_today():
    return build_readout(date.today())


@router.get("/curiosity-compactions")
def curiosity_compactions(limit: int = Query(default=20, ge=1, le=100)):
    """Stored summaries/provenance, including pending and explicitly empty windows."""
    from app.curiosity_compaction_store import recent
    try:
        return {"items": recent(limit)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Curiosity compaction store unavailable") from exc
