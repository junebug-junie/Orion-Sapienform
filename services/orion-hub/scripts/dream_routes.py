"""Read-only Dream operator API. Never claims hypotheses or triggers sleep.

SQL contracts: manual_migration_dream_cycle_v2.sql. Readiness is owned by
orion-dream's /dreams/cycle/pressure; scoring reuses the waking offer contract.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import create_engine, text

from app.settings import settings
from orion.curiosity.worldview import WorldviewReader
from orion.dream.hypotheses import score_hypotheses
from orion.reverie.visual_storage import SUPPORTED_MIMES, load_visual_artifact, sniff_image
from orion.schemas.dream_cycle import SleepPressureV1

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dream", tags=["dream"])
_engine_instance: Any = None
SCORE_LIMIT = 10000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Carried dreams (orion-dream app/carry.py finish): dream.result.v1 with profile dream.carry,
# one fragment per hop. The dreams table is the only record; no carry table exists.
CARRY_PROFILE = "dream.carry"


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


@router.get("/carries")
def carries(response: Response, limit: int = Query(6, ge=1, le=20)) -> dict:
    """Recent carried dreams, hops in order: passage, picture, what Orion saw, ..."""
    _no_store(response)
    rows = _rows(
        "SELECT id, created_at, tldr, fragments, metrics->'_dream_audit' AS audit FROM dreams "
        "WHERE metrics->'_dream_audit'->>'profile' = :profile ORDER BY created_at DESC, id DESC LIMIT :limit",
        {"profile": CARRY_PROFILE, "limit": limit},
    )
    out = []
    for row in rows:
        audit = row["audit"] if isinstance(row["audit"], dict) else {}
        trigger = audit.get("trigger") if isinstance(audit.get("trigger"), dict) else {}
        hops = [f for f in (row["fragments"] or []) if isinstance(f, dict) and f.get("kind") in ("text", "image")]
        hops.sort(key=lambda f: int(f.get("index") or 0))
        out.append({
            "id": row["id"], "created_at": row["created_at"], "tldr": row["tldr"],
            "trigger_id": trigger.get("trigger_id"), "stopped_reason": trigger.get("stopped_reason"),
            "sleep_cycle_id": (trigger.get("sleep") or {}).get("cycle_id") if isinstance(trigger.get("sleep"), dict) else None,
            "hops": [{k: f.get(k) for k in ("index", "kind", "passage", "image_prompt", "sha256", "caption")} for f in hops],
        })
    return {"carries": out}


def _carry_image_known(sha256: str) -> bool:
    # Only pictures a carried dream names: the route is not a general file reader.
    return bool(_rows(
        "SELECT 1 FROM dreams WHERE metrics->'_dream_audit'->>'profile' = :profile "
        "AND fragments @> CAST(:needle AS jsonb) LIMIT 1",
        {"profile": CARRY_PROFILE, "needle": f'[{{"sha256": "{sha256}"}}]'},
    ))


@router.get("/carry/image/{sha256}")
async def carry_image(sha256: str) -> Response:
    if not _SHA256_RE.match(sha256):
        raise HTTPException(400, "invalid artifact id")
    if not await asyncio.to_thread(_carry_image_known, sha256):
        raise HTTPException(404, "dream picture not found")
    try:
        data = await asyncio.to_thread(load_visual_artifact, sha256, base_dir=settings.REVERIE_VISUAL_STORAGE_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(404, "dream picture missing on disk") from exc
    if hashlib.sha256(data).hexdigest() != sha256:
        logger.error("dream carry image hash mismatch sha=%s", sha256[:12])
        raise HTTPException(500, "artifact integrity check failed")
    sniffed = sniff_image(data)
    mime = sniffed[0] if sniffed and sniffed[0] in SUPPORTED_MIMES else "application/octet-stream"
    return Response(content=data, media_type=mime, headers={
        "Cache-Control": "public, max-age=31536000, immutable",
        "Content-Security-Policy": "default-src 'none'; sandbox",
        "X-Content-Type-Options": "nosniff",
    })
