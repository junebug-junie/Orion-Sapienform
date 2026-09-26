"""Read-only Dream operator API. Never claims hypotheses or triggers sleep.

SQL contracts: manual_migration_dream_cycle_v2.sql. Readiness is owned by
orion-dream's /dreams/cycle/pressure; scoring reuses the waking offer contract.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import create_engine, text

from app.settings import settings
from orion.curiosity.worldview import WorldviewReader
from orion.dream.hypotheses import score_hypotheses
from orion.schemas.dream_cycle import SleepPressureV1

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dream", tags=["dream"])
_engine_instance: Any = None
SCORE_LIMIT = 10000


def _engine():
    global _engine_instance
    if _engine_instance is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            raise HTTPException(503, "dream_database_not_configured")
        _engine_instance = create_engine(
            uri, pool_pre_ping=True,
            connect_args={"connect_timeout": 5, "options": "-c statement_timeout=8000"},
        )
    return _engine_instance


def _rows(sql: str, params: dict | None = None) -> list[dict]:
    try:
        with _engine().connect() as conn:
            return [dict(row) for row in conn.execute(text(sql), params or {}).mappings().all()]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Dream database read failed: %s", type(exc).__name__)
        raise HTTPException(503, "dream_database_unavailable") from exc


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


@router.get("/pressure")
async def pressure(response: Response) -> dict:
    _no_store(response)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.get(
                settings.HUB_DREAM_SERVICE_URL.rstrip("/") + "/dreams/cycle/pressure"
            )
            result.raise_for_status()
            data = result.json()
        reading = SleepPressureV1.model_validate(data["pressure"])
        if any(type(data[key]) is not bool for key in ("enabled", "too_soon", "should_sleep", "is_idle")):
            raise ValueError("invalid readiness flags")
        return {
            **data,
            "pressure": reading.model_dump(mode="json"),
            "ready": data["enabled"] and data["should_sleep"] and not data["too_soon"],
            "offer_enabled": settings.HUB_CURIOSITY_DREAM_HYPOTHESES_ENABLED,
        }
    except Exception as exc:
        logger.warning("Dream pressure unavailable: %s", type(exc).__name__)
        raise HTTPException(503, "dream_pressure_unavailable") from exc


@router.get("/cycles")
def cycles(
    response: Response,
    limit: int = Query(12, ge=1, le=50),
    before: datetime | None = None,
    before_id: str | None = Query(None, max_length=100),
) -> dict:
    _no_store(response)
    if (before is None) != (before_id is None):
        raise HTTPException(422, "before and before_id must be supplied together")
    where = "WHERE (started_at, cycle_id) < (:before, :before_id)" if before else ""
    rows = _rows(
        "SELECT cycle_id, trigger, status, started_at, ended_at, pressure, replay_count, "
        "hypothesis_count, no_link_count, unparseable_count, llm_failures, compaction_delta_id "
        f"FROM dream_cycle {where} ORDER BY started_at DESC, cycle_id DESC LIMIT :limit",
        {"before": before, "before_id": before_id, "limit": limit + 1},
    )
    page = rows[:limit]
    return {
        "cycles": page, "has_more": len(rows) > limit,
        "next_cursor": {"before": page[-1]["started_at"], "before_id": page[-1]["cycle_id"]} if page else None,
    }


@router.get("/cycles/{cycle_id}")
def cycle_detail(cycle_id: str, response: Response) -> dict:
    _no_store(response)
    rows = _rows("SELECT cycle_json FROM dream_cycle WHERE cycle_id = :id", {"id": cycle_id})
    if not rows:
        raise HTTPException(404, "dream_cycle_not_found")
    # Current offer state is deliberately read from the hypothesis table, not
    # the immutable cycle snapshot. SELECT does not consume the waking offer.
    hypotheses = _rows(
        "SELECT hypothesis_id, arm, claim, why, ref_a, ref_b, created_at, expires_at, "
        "offered_at, offered_run_id, (expires_at <= now()) AS expired "
        "FROM dream_hypothesis WHERE cycle_id = :id ORDER BY created_at, hypothesis_id",
        {"id": cycle_id},
    )
    return {"cycle": rows[0]["cycle_json"], "hypotheses": hypotheses}


def _prior_rows() -> list[dict]:
    reader = WorldviewReader(
        host=settings.HUB_CURIOSITY_GRAPH_HOST,
        port=settings.HUB_CURIOSITY_GRAPH_PORT,
        graph_name=settings.HUB_CURIOSITY_GRAPH_OWN,
    )
    return reader.query(
        "MATCH (p:Prior) WHERE p.formed_from STARTS WITH 'dream_hypothesis:' "
        "RETURN p.prior_id AS prior_id, p.formed_from AS formed_from, "
        f"p.status AS status, p.times_tested AS times_tested LIMIT {SCORE_LIMIT + 1}"
    )


@router.get("/scorecard")
def scorecard(response: Response) -> dict:
    _no_store(response)
    offered = _rows(
        "SELECT hypothesis_id, arm FROM dream_hypothesis WHERE offered_at IS NOT NULL "
        f"LIMIT {SCORE_LIMIT + 1}"
    )
    try:
        priors = _prior_rows()
    except Exception as exc:
        logger.warning("Dream worldview read failed: %s", type(exc).__name__)
        raise HTTPException(503, "dream_worldview_unavailable") from exc
    if len(offered) > SCORE_LIMIT or len(priors) > SCORE_LIMIT:
        raise HTTPException(503, "dream_scorecard_limit_exceeded")
    return score_hypotheses(offered, priors).as_dict()
