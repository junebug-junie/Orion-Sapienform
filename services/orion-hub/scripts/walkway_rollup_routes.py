"""Walkway history panel: the walkway rollups by hour, day, week and month (2026-10-18).

``orion-sql-writer`` (app/vision_rollup_loop.py) writes one ``vision_rollup`` row per
(stream, grain, period): deterministic counts plus one plain-English line
(orion/vision/walkway_rollup.py). This route reads them for the Hub's "Walkway
history" panel under the "Walkway (live)" picture (static/js/walkway-history.js).

- ``GET /api/vision/rollups?stream_id=walkway&grain=hour|day|week|month&limit=N``:
  newest first. Each row carries its line and the parts the panel shows (totals by
  kind, named people, regulars, busiest/quietest, camera minutes and down hours,
  captions, stories only when the spot-check gate passed). The per-individual maps
  (individual ids) are never sent.

Read-only. Same asyncpg pool as the ask card (``app.state.memory_pg_pool``); a missing
table (migration not applied) is a 503 ``rollup_schema_missing``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from scripts.ask_routes import _iso, _pool, _raise_store_http

router = APIRouter(tags=["walkway-rollups"])

GRAINS = ("hour", "day", "week", "month")
MAX_LIMIT = {"hour": 72, "day": 62, "week": 26, "month": 24}
_STREAM_RE = re.compile(r"[a-z0-9_-]{1,32}")
_PANEL_KEYS = ("total", "kinds", "named", "regulars", "busiest", "quietest", "baseline", "camera", "captions",
               "stories", "excluded", "children", "covered_until", "local")

_LIST_SQL = """
SELECT stream_id, grain, period_start, period_end, label, complete, line, rollup, computed_at
FROM vision_rollup
WHERE stream_id = $1 AND grain = $2
ORDER BY period_start DESC
LIMIT $3
"""


def _raise_rollup_http(exc: BaseException) -> None:
    try:
        _raise_store_http(exc)
    except HTTPException as http:
        detail = str(http.detail).replace("ask_", "rollup_", 1)
        raise HTTPException(status_code=http.status_code, detail=detail) from exc


def panel_row(row: Any) -> dict[str, Any]:
    rollup = row["rollup"]
    if isinstance(rollup, str):
        try:
            rollup = json.loads(rollup)
        except ValueError:
            rollup = {}
    rollup = rollup if isinstance(rollup, dict) else {}
    parts = {k: rollup.get(k) for k in _PANEL_KEYS if k in rollup}
    named = parts.get("named") or {}
    parts["named"] = {name: {k: v for k, v in (entry or {}).items() if k != "individual_ids"}
                      for name, entry in named.items()}
    reg = parts.get("regulars") or {}
    parts["regulars"] = {"count": reg.get("count", 0), "by_kind": reg.get("by_kind", {})}
    stories = parts.get("stories") or {}
    if not stories.get("gate_passed"):
        parts["stories"] = {"gate_passed": False, "checked": stories.get("checked", 0)}
    return {
        "stream_id": row["stream_id"], "grain": row["grain"], "label": row["label"],
        "period_start": _iso(row["period_start"]), "period_end": _iso(row["period_end"]),
        "complete": bool(row["complete"]), "line": row["line"], "computed_at": _iso(row["computed_at"]),
        **parts,
    }


@router.get("/api/vision/rollups")
async def list_rollups(
    request: Request,
    stream_id: str = Query("walkway"),
    grain: str = Query("day"),
    limit: int = Query(0, ge=0),
) -> dict[str, Any]:
    if grain not in GRAINS:
        raise HTTPException(status_code=400, detail="bad_grain")
    if not _STREAM_RE.fullmatch(stream_id or ""):
        raise HTTPException(status_code=400, detail="bad_stream_id")
    n = min(limit or {"hour": 24, "day": 14, "week": 8, "month": 6}[grain], MAX_LIMIT[grain])
    pool = _pool(request)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(_LIST_SQL, stream_id, grain, n)
    except Exception as exc:
        _raise_rollup_http(exc)
    return {"ok": True, "stream_id": stream_id, "grain": grain, "rows": [panel_row(r) for r in rows]}
