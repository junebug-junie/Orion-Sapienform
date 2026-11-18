"""What Claude looked at, and what Orion allows: Hub's read-only view of Claude's tools (Patch 1b).

Design: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md section 3, "The audit
trail Juniper sees". Endpoints first; a panel can come later.

- ``GET /api/claude-tools/audit``: recent rows of ``claude_tool_call_log`` (one per phase of each call:
  ask, answer, result), newest first, plus the consent summary per category over ``days``
  (orion/claude_tools/report.py: yes rate, unanswered rate, reason share, latency p50/p90, flips, and
  the rubber-stamp flag).
- ``GET /api/claude-tools/grants``: Orion's standing setting per category, with its reason
  (``$ORION_DIG_HOME/self/claude_grants.jsonl``; a category never set reads ``ask``).

Read-only. A database that cannot be read answers 503 with the reason, never an empty list.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("orion-hub.claude_tools")

router = APIRouter(tags=["claude-tools"])

_COLUMNS = (
    "event_id, phase, call_id, turn_request_id, room_id, claude_session_id, hub_session_id, trigger, tool, "
    "category, arguments, reason, setting, decision_path, shared, orion_answer, orion_scope, orion_reason, "
    "consent_path, consent_latency_ms, act_id, result_chars, error, created_at"
)


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return None
    from scripts.pg_engine import module_engine

    return module_engine(__name__, uri)


def _jsonable(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items()}


def fetch_audit(*, limit: int, days: float, category: Optional[str], turn: Optional[str]) -> Dict[str, Any]:
    """Sync; run in a thread. Raises when the table cannot be read."""
    engine = _engine()
    if engine is None:
        raise RuntimeError("POSTGRES_URI is not set")
    from sqlalchemy import text

    where = ["created_at >= now() - make_interval(secs => :secs)"]
    params: Dict[str, Any] = {"secs": float(days) * 86400.0, "lim": int(limit)}
    if category:
        where.append("category = :cat")
        params["cat"] = category
    if turn:
        where.append("turn_request_id = :turn")
        params["turn"] = turn
    sql = f"SELECT {_COLUMNS} FROM claude_tool_call_log WHERE {' AND '.join(where)} ORDER BY created_at DESC LIMIT :lim"
    answers_sql = (f"SELECT {_COLUMNS} FROM claude_tool_call_log WHERE phase = 'answer' AND "
                   "created_at >= now() - make_interval(secs => :secs) ORDER BY created_at")
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(text(sql), params).mappings().all()]
        answers = [dict(r) for r in conn.execute(text(answers_sql), {"secs": params["secs"]}).mappings().all()]
    from orion.claude_tools.report import consent_summary

    return {
        "days": days,
        "rows": [_jsonable(r) for r in rows],
        "count": len(rows),
        "consent_summary": consent_summary(answers),
    }


@router.get("/api/claude-tools/audit")
async def claude_tools_audit(
    limit: int = Query(100, ge=1, le=500),
    days: float = Query(7.0, gt=0, le=90),
    category: Optional[str] = Query(None, max_length=40),
    turn: Optional[str] = Query(None, max_length=120),
) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(fetch_audit, limit=limit, days=days, category=category, turn=turn)
    except Exception as exc:  # noqa: BLE001 -- unreadable is not empty
        logger.warning("claude_tools_audit_failed", exc_info=True)
        raise HTTPException(status_code=503, detail=f"claude_tool_call_log unreadable: {type(exc).__name__}") from exc


@router.get("/api/claude-tools/grants")
async def claude_tools_grants() -> Dict[str, Any]:
    from orion.claude_tools.grants import grants_path, grants_view

    try:
        return await asyncio.to_thread(grants_view, grants_path())
    except Exception as exc:  # noqa: BLE001
        logger.warning("claude_tools_grants_failed", exc_info=True)
        raise HTTPException(status_code=503, detail=f"grants unreadable: {type(exc).__name__}") from exc


__all__ = ["fetch_audit", "router"]
