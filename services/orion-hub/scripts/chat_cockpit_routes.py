"""Cockpit turn sighting timeline read API for Soft HUD rewind.

Append-only hops land in ``cockpit_turn_sighting`` via orion-sql-writer
consuming ``orion:cockpit:hop``. This module is a pure read-side projection:
ordered hops plus explicit canonical-stage gaps for the scrubber.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, get_args

from fastapi import APIRouter
from sqlalchemy import create_engine, text

from orion.schemas.cockpit_sighting import CockpitHopV1, CockpitStageV1

logger = logging.getLogger("orion-hub.chat_cockpit")

router = APIRouter(prefix="/api/chat/turn", tags=["chat-cockpit"])

CANONICAL_STAGES: tuple[str, ...] = get_args(CockpitStageV1)

_engine = None


def _postgres_engine():
    """Shared engine for the conjourney Postgres DB -- same POSTGRES_URI
    already used by sibling substrate_*_routes.py and chat_turn_trace_routes.
    """
    global _engine
    if _engine is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            return None
        _engine = create_engine(uri, pool_pre_ping=True)
    return _engine


def _load_hops(correlation_id: str) -> list[dict[str, Any]]:
    """Load persisted hops for one turn, ordered by seq. Degrades to []."""
    engine = _postgres_engine()
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT correlation_id, seq, ts, stage, visor_line, status,
                               summary, raw, producer
                        FROM cockpit_turn_sighting
                        WHERE correlation_id = :correlation_id
                        ORDER BY seq
                        """
                    ),
                    {"correlation_id": correlation_id},
                )
                .mappings()
                .all()
            )
    except Exception:
        logger.warning("chat_cockpit cockpit_turn_sighting query failed", exc_info=True)
        return []

    hops: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["schema_version"] = "cockpit.hop.v1"
        if payload.get("ts") is not None:
            payload["ts"] = str(payload["ts"])
        for key in ("summary", "raw"):
            val = payload.get(key)
            if isinstance(val, str):
                payload[key] = json.loads(val)
        try:
            hop = CockpitHopV1.model_validate(payload)
        except Exception:
            logger.warning("chat_cockpit hop row failed schema validation", exc_info=True)
            continue
        hops.append(hop.model_dump(mode="json"))
    return hops


async def get_cockpit_timeline(correlation_id: str) -> dict[str, Any]:
    corr = str(correlation_id or "").strip()
    hops = await asyncio.to_thread(_load_hops, corr)
    hops = sorted(hops, key=lambda h: h["seq"])
    present_stages = {h["stage"] for h in hops}
    gaps = [stage for stage in CANONICAL_STAGES if stage not in present_stages]
    return {
        "correlation_id": corr,
        "hops": hops,
        "complete": bool(hops),  # Slice A: terminal marker deferred; non-empty = complete
        "gaps": gaps,
    }


@router.get("/{correlation_id}/cockpit")
async def api_chat_turn_cockpit(correlation_id: str) -> dict[str, Any]:
    """Cockpit timeline lookup: never 404s on empty -- missing hops are gaps."""
    return await get_cockpit_timeline(correlation_id)
