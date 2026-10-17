"""Walkway stories spot-check card (motion stories slice 1, 2026-10-16).

docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md.
``orion-sql-writer`` (app/vision_motion_loop.py) writes one row per passing
thing into ``vision_motion_story``: a plain sentence plus the boxes it rests
on. Juniper decided (2026-10-16) that stories stay HUB ONLY until she has
spot-checked them: at least ``GATE_MIN_CHECKED`` checked, at least
``GATE_MIN_RIGHT`` fully right, and zero wrong names. This module is that
spot-check:

- ``GET /api/vision/motion-stories`` lists recent published stories (clean
  frames by default) with their evidence boxes and thumbnails, and the
  running tally against the gate.
- ``POST /api/vision/motion-stories/{story_id}/verdict`` records Right /
  Wrong name / Wrong motion / Wrong other in ``vision_motion_story_verdict``.
  One verdict per story; a new click replaces the old one. "Can't tell"
  (2026-10-17: no picture, or too dark) is stored as ``verdict NULL,
  cant_tell true``: the story leaves the unchecked queue but never counts as
  right or wrong.
- ``GET /api/vision/motion-stories/{story_id}/strip/{name}`` serves the
  story's pictures of the moment (``crop_0..3.jpg``, ``frame.jpg``) that
  orion-sql-writer saved while the frames still existed
  (app/vision_story_strip.py). ``checkable_only`` (default on) lists only
  stories with a saved frame that is not dark, so the queue is all judgeable.

The tally counts one rules version at a time (the newest one with stories):
a verdict on old rules never vouches for new ones. No bus event: nothing
consumes a verdict yet, and the table is the only truth.

Same asyncpg pool as the ask card (``app.state.memory_pg_pool``); all DB calls
are awaited (``scripts/check_async_routes_not_blocking.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel

from scripts.ask_routes import _iso, _pool, _raise_store_http, _read_thumb


def _raise_motion_http(exc: BaseException) -> None:
    """The ask card's store-error mapping (schema missing / transport -> 503), with this
    card's own detail names so a 503 here is not mistaken for the ask table."""
    try:
        _raise_store_http(exc)
    except HTTPException as http:
        detail = str(http.detail).replace("ask_", "motion_story_", 1)
        raise HTTPException(status_code=http.status_code, detail=detail) from exc

logger = logging.getLogger("orion-hub.motion_stories")

router = APIRouter(tags=["motion-stories"])

GATE_MIN_CHECKED = 50
GATE_MIN_RIGHT = 0.80
GATE_MAX_WRONG_NAMES = 0
VERDICTS = ("right", "wrong_name", "wrong_motion", "wrong_other")
CANT_TELL = "cant_tell"
MAX_LIST = 50
_STORY_ID_RE = re.compile(r"[0-9a-f]{32}")
# The only file names sql-writer writes into a strip directory.
_STRIP_NAME_RE = re.compile(r"(?:crop_[0-3]|frame)\.jpg")
DEFAULT_STRIP_DIR = "/mnt/telemetry/vision/story_strips"

_LIST_SQL = """
SELECT s.story_id, s.stream_id, s.kind, s.started_at, s.ended_at, s.detection_count, s.used_detections,
       s.speed_class, s.zone_path, s.dir_lateral, s.dir_depth, s.individual_id, s.name_label,
       s.companion_kinds, s.clean, s.sentence, s.evidence, s.rules_version,
       CASE WHEN v.cant_tell THEN 'cant_tell' ELSE v.verdict END AS verdict,
       s.evidence_strip, s.frame_luminance, s.scene_luminance, s.scene_contrast, s.dark
FROM vision_motion_story s
LEFT JOIN vision_motion_story_verdict v ON v.story_id = s.story_id
WHERE s.published AND ($1::boolean IS FALSE OR s.clean)
  AND ($2::boolean IS FALSE OR v.story_id IS NULL)
  AND ($4::integer IS NULL OR s.rules_version = $4)
  AND ($5::boolean IS FALSE OR (s.dark IS FALSE AND s.evidence_strip->'frame'->>'file' IS NOT NULL
                                AND NOT (s.evidence_strip ? 'pruned_at')))
ORDER BY s.started_at DESC
LIMIT $3
"""

_TALLY_SQL = """
SELECT CASE WHEN v.cant_tell THEN 'cant_tell' ELSE v.verdict END AS verdict, count(*) AS n,
       count(*) FILTER (WHERE s.name_label IS NOT NULL) AS named
FROM vision_motion_story_verdict v
JOIN vision_motion_story s ON s.story_id = v.story_id
WHERE s.published AND s.clean AND s.rules_version = $1
GROUP BY 1
"""


class VerdictBody(BaseModel):
    verdict: Literal["right", "wrong_name", "wrong_motion", "wrong_other", "cant_tell"]


def tally_from_counts(counts: dict[str, int], named_checked: int, rules_version: int | None) -> dict[str, Any]:
    checked = sum(counts.get(v, 0) for v in VERDICTS)
    right = counts.get("right", 0)
    wrong_names = counts.get("wrong_name", 0)
    pct = right / checked if checked else None
    passed = (checked >= GATE_MIN_CHECKED and pct is not None and pct >= GATE_MIN_RIGHT
              and wrong_names <= GATE_MAX_WRONG_NAMES)
    return {
        "rules_version": rules_version,
        "checked": checked,
        "right": right,
        "wrong_name": wrong_names,
        "wrong_motion": counts.get("wrong_motion", 0),
        "wrong_other": counts.get("wrong_other", 0),
        "named_checked": named_checked,
        # Juniper could not judge these (no picture, too dark): never right, never wrong.
        "cant_tell": counts.get(CANT_TELL, 0),
        "pct_right": round(pct, 4) if pct is not None else None,
        "gate": {"min_checked": GATE_MIN_CHECKED, "min_right": GATE_MIN_RIGHT,
                 "max_wrong_names": GATE_MAX_WRONG_NAMES, "passed": passed},
    }


async def _tally(conn: Any) -> dict[str, Any]:
    version = await conn.fetchval(
        "SELECT max(rules_version) FROM vision_motion_story WHERE published AND clean")
    if version is None:
        return tally_from_counts({}, 0, None)
    rows = await conn.fetch(_TALLY_SQL, version)
    counts = {r["verdict"]: int(r["n"]) for r in rows if r["verdict"]}
    named = sum(int(r["named"]) for r in rows if r["verdict"] in VERDICTS)
    return tally_from_counts(counts, named, int(version))


def _story_dict(row: Any) -> dict[str, Any]:
    out = dict(row)
    ev = out.get("evidence")
    if isinstance(ev, str):
        try:
            ev = json.loads(ev)
        except ValueError:
            ev = []
    out["evidence"] = list(ev or [])
    strip = out.get("evidence_strip")
    if isinstance(strip, str):
        try:
            strip = json.loads(strip)
        except ValueError:
            strip = None
    out["evidence_strip"] = strip if isinstance(strip, dict) else None
    for key in ("zone_path", "companion_kinds"):
        out[key] = list(out.get(key) or [])
    for key in ("started_at", "ended_at"):
        out[key] = _iso(out.get(key))
    return out


@router.get("/api/vision/motion-stories")
async def list_motion_stories(
    request: Request,
    limit: int = Query(20, ge=1, le=MAX_LIST),
    clean_only: bool = Query(True),
    unchecked_only: bool = Query(False),
    checkable_only: bool = Query(True),
) -> dict[str, Any]:
    pool = _pool(request)
    try:
        async with pool.acquire() as conn:
            # Only stories under the rules the tally counts: a verdict on an
            # older rules version would never count toward the gate.
            tally = await _tally(conn)
            rows = await conn.fetch(_LIST_SQL, clean_only, unchecked_only, limit, tally["rules_version"],
                                    checkable_only)
    except Exception as exc:
        _raise_motion_http(exc)
    return {"ok": True, "stories": [_story_dict(r) for r in rows], "tally": tally}


@router.post("/api/vision/motion-stories/{story_id}/verdict")
async def record_motion_story_verdict(story_id: str, body: VerdictBody, request: Request) -> dict[str, Any]:
    if not _STORY_ID_RE.fullmatch(story_id or ""):
        raise HTTPException(status_code=400, detail="bad_story_id")
    pool = _pool(request)
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                version = await conn.fetchval(
                    "SELECT rules_version FROM vision_motion_story WHERE story_id = $1 AND published", story_id)
                if version is None:
                    raise HTTPException(status_code=404, detail="story_not_found")
                await conn.execute(
                    """
                    INSERT INTO vision_motion_story_verdict (story_id, verdict, rules_version, cant_tell)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (story_id) DO UPDATE
                      SET verdict = EXCLUDED.verdict, rules_version = EXCLUDED.rules_version,
                          cant_tell = EXCLUDED.cant_tell, updated_at = now()
                    """,
                    story_id, None if body.verdict == CANT_TELL else body.verdict, version,
                    body.verdict == CANT_TELL,
                )
            tally = await _tally(conn)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_motion_http(exc)
    logger.info("motion_story_verdict story_id=%s verdict=%s", story_id, body.verdict)
    return {"ok": True, "story_id": story_id, "verdict": body.verdict, "tally": tally}


def _strip_dir() -> Path:
    try:
        from scripts.settings import settings

        raw = str(getattr(settings, "HUB_VISION_STORY_STRIP_DIR", "") or "")
    except Exception:  # settings env not loaded (bare test process)
        raw = os.getenv("HUB_VISION_STORY_STRIP_DIR", "")
    return Path(raw.strip() or DEFAULT_STRIP_DIR)


def _read_strip(root: Path, story_id: str, name: str) -> bytes | None:
    """The story directory must be a real directory, not a link out of the root."""
    story_dir = root / story_id
    if story_dir.is_symlink():
        return None
    return _read_thumb(story_dir / name)


@router.get("/api/vision/motion-stories/{story_id}/strip/{name}")
async def get_motion_story_strip(story_id: str, name: str) -> Response:
    """One picture of a story's moment. Both path parts are whitelisted by pattern
    (32-hex story id; crop_0..3.jpg or frame.jpg), so no traversal is possible;
    the read refuses links, FIFOs and oversize files (same reader as the crop thumbs).
    404 when absent (not saved, frame was gone, or deleted after 30 days)."""
    if not _STORY_ID_RE.fullmatch(story_id or "") or not _STRIP_NAME_RE.fullmatch(name or ""):
        raise HTTPException(status_code=400, detail="bad_strip_path")
    data = await asyncio.to_thread(_read_strip, _strip_dir(), story_id, name)
    if data is None:
        raise HTTPException(status_code=404, detail="strip_not_found")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400", "X-Content-Type-Options": "nosniff"},
    )
