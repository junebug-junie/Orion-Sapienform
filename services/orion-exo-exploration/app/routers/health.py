from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.settings import settings
from app.storage.pg import pg_conn

logger = logging.getLogger("orion-exo-exploration.health")

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    return {"ok": True, "service": settings.service_name, "version": settings.service_version}


@router.get("/readyz")
def readyz():
    try:
        with pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"ok": True, "postgres": "reachable"}
    except Exception as exc:  # noqa: BLE001 -- readiness must report, never 500
        logger.warning("exo_exploration_readyz_pg_unreachable err=%s", exc)
        return JSONResponse(status_code=503, content={"ok": False, "postgres": "unreachable", "error": str(exc)[:200]})
