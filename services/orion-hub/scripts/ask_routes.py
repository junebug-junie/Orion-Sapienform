"""Orion's open questions to Juniper, and her answers (OrionAskV1).

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 3.
``orion-sql-writer``'s individuals loop opens asks (inserts ``orion_ask`` rows). This module lists open asks for the Vision panel card
and records Juniper's answer or dismissal:

1. ``UPDATE orion_ask ... WHERE status='open'`` -- only an open, unexpired
   row moves; anything else is a 409, so a double click or a stale tab cannot
   overwrite an answer.
2. Publish ``OrionAskAnsweredV1`` on ``orion:ask:answered`` (consumed by
   ``orion-substrate-runtime``, which writes the substrate entity).

The row is the source of truth. ``orion-sql-writer`` applies answers to ``vision_individual.label`` by reading
``status='answered' AND applied_at IS NULL`` on its own clock, so a failed
publish does not lose the label -- it is reported as ``published: false`` and
logged. It does lose the substrate entity for that answer: nothing replays
``orion:ask:answered`` today (known gap).

There is no "ask opened" bus event: the Hub has no push channel for panels,
so the card polls ``GET /api/asks`` and Postgres stays the only truth.

``GET /api/vision/crop-thumbs/{sha256}`` serves the ask card's picture: a
small JPEG of one embedded crop, written by orion-vision-host (never for a
no-embed-zone box) into a directory this Hub mounts read-only
(``HUB_VISION_CROP_THUMB_DIR``). The only caller-supplied component is a
64-char lowercase hex digest -- no path to traverse, nothing to enumerate --
and the bytes are re-hashed before they are served.

Uses the Hub's asyncpg pool (``app.state.memory_pg_pool``, same ``conjourney``
database sql-writer writes to). All DB calls are awaited, so nothing blocks the
event loop (``scripts/check_async_routes_not_blocking.py``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.ask import OrionAskAnsweredV1

try:
    from asyncpg.exceptions import (
        InterfaceError as _AsyncpgInterfaceError,
        PostgresConnectionError as _AsyncpgConnectionError,
        UndefinedTableError as _AsyncpgUndefinedTableError,
    )

    _TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
        TimeoutError,
        OSError,
        _AsyncpgConnectionError,
        _AsyncpgInterfaceError,
    )
except ImportError:  # pragma: no cover - optional in minimal dev envs
    _AsyncpgUndefinedTableError = None  # type: ignore[misc, assignment]
    _TRANSPORT_ERRORS = (TimeoutError, OSError)

logger = logging.getLogger("orion-hub.asks")

router = APIRouter(tags=["asks"])

CHANNEL_ASK_ANSWERED = "orion:ask:answered"
ASK_ANSWERED_KIND = "orion.ask.answered.v1"
MAX_ANSWER_CHARS = 500
MAX_LIST = 50

DEFAULT_CROP_THUMB_DIR = "/mnt/telemetry/orion-vision-host/crop_thumbs"
_THUMB_ID_RE = re.compile(r"[0-9a-f]{64}")
# Thumbnails are tiny (<= 160 px); anything larger is not one of ours.
MAX_THUMB_BYTES = 256 * 1024

_SELECT_COLUMNS = (
    "ask_id, asked_of, question, evidence_refs, image_ref, status, answer, "
    "answered_at, created_at, expires_at, source_kind, source_ref"
)


class AskAnswerBody(BaseModel):
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)


def _pool(request: Request):
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="postgres_pool_unavailable")
    return pool


def _raise_store_http(exc: BaseException) -> None:
    if _AsyncpgUndefinedTableError is not None and isinstance(exc, _AsyncpgUndefinedTableError):
        logger.warning("ask_schema_missing error=%s", exc)
        raise HTTPException(status_code=503, detail="ask_schema_missing") from exc
    if isinstance(exc, _TRANSPORT_ERRORS):
        logger.warning("ask_store_transport error=%s", exc)
        raise HTTPException(status_code=503, detail="ask_store_unavailable") from exc
    raise exc


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return value


def _row_dict(row: Any) -> dict[str, Any]:
    out = dict(row)
    refs = out.get("evidence_refs")
    if isinstance(refs, str):
        try:
            refs = json.loads(refs)
        except ValueError:
            refs = []
    out["evidence_refs"] = list(refs or [])
    for key in ("answered_at", "created_at", "expires_at"):
        out[key] = _iso(out.get(key))
    return out


def _source_ref() -> ServiceRef:
    try:
        from scripts.settings import settings

        return ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME)
    except Exception:  # settings env not loaded (bare test process)
        return ServiceRef(name="hub", version="0.3.0", node="athena")


def build_answered_envelope(event: OrionAskAnsweredV1) -> BaseEnvelope:
    return BaseEnvelope(
        kind=ASK_ANSWERED_KIND,
        source=_source_ref(),
        payload=event.model_dump(mode="json"),
    )


async def _publish_answered(event: OrionAskAnsweredV1) -> bool:
    from .main import bus

    if bus is None or not getattr(bus, "enabled", True):
        logger.warning("ask_answered_publish_skipped ask_id=%s reason=bus_unavailable", event.ask_id)
        return False
    try:
        await bus.publish(CHANNEL_ASK_ANSWERED, build_answered_envelope(event))
        return True
    except Exception as exc:
        logger.warning("ask_answered_publish_failed ask_id=%s error=%s", event.ask_id, exc)
        return False


@router.get("/api/asks")
async def list_asks(
    request: Request,
    status: Literal["open", "answered", "dismissed", "expired"] = Query("open"),
    limit: int = Query(20, ge=1, le=MAX_LIST),
) -> dict[str, Any]:
    pool = _pool(request)
    # "open" means still answerable: an open row past expires_at is not shown,
    # because answering it would 409 anyway.
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM orion_ask
                WHERE status = $1
                  AND ($1 <> 'open' OR expires_at IS NULL OR expires_at > now())
                ORDER BY created_at DESC
                LIMIT $2
                """,
                status,
                limit,
            )
    except Exception as exc:
        _raise_store_http(exc)
    return {"ok": True, "status": status, "asks": [_row_dict(r) for r in rows]}


async def _close_ask(
    request: Request,
    ask_id: str,
    *,
    new_status: Literal["answered", "dismissed"],
    answer: Optional[str],
) -> dict[str, Any]:
    pool = _pool(request)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE orion_ask
                SET status = $2, answer = $3, answered_at = now()
                WHERE ask_id = $1
                  AND status = 'open'
                  AND (expires_at IS NULL OR expires_at > now())
                RETURNING ask_id, status, answer, answered_at, source_kind, source_ref
                """,
                ask_id,
                new_status,
                answer,
            )
            if row is None:
                existing = await conn.fetchrow(
                    "SELECT status, expires_at FROM orion_ask WHERE ask_id = $1",
                    ask_id,
                )
    except Exception as exc:
        _raise_store_http(exc)

    if row is None:
        if existing is None:
            raise HTTPException(status_code=404, detail="ask_not_found")
        current = existing["status"]
        detail = f"ask_not_open:{current}"
        if current == "open":
            detail = "ask_not_open:expired"
        raise HTTPException(status_code=409, detail=detail)

    answered_at = row["answered_at"]
    if isinstance(answered_at, datetime) and answered_at.tzinfo is None:
        answered_at = answered_at.replace(tzinfo=timezone.utc)
    event = OrionAskAnsweredV1(
        ask_id=row["ask_id"],
        status=row["status"],
        answer=row["answer"],
        answered_at=answered_at or datetime.now(timezone.utc),
        source_kind=row["source_kind"],
        source_ref=row["source_ref"],
    )
    published = await _publish_answered(event)
    logger.info(
        "ask_closed ask_id=%s status=%s source_kind=%s source_ref=%s published=%s",
        event.ask_id,
        event.status,
        event.source_kind,
        event.source_ref,
        published,
    )
    return {"ok": True, "ask": event.model_dump(mode="json"), "published": published}


@router.post("/api/asks/{ask_id}/answer")
async def answer_ask(request: Request, ask_id: str, body: AskAnswerBody) -> dict[str, Any]:
    answer = " ".join(body.answer.split())
    if not answer:
        raise HTTPException(status_code=422, detail="answer_empty")
    return await _close_ask(request, ask_id, new_status="answered", answer=answer)


@router.post("/api/asks/{ask_id}/dismiss")
async def dismiss_ask(request: Request, ask_id: str) -> dict[str, Any]:
    return await _close_ask(request, ask_id, new_status="dismissed", answer=None)


def _crop_thumb_dir() -> Path:
    try:
        from scripts.settings import settings

        raw = str(getattr(settings, "HUB_VISION_CROP_THUMB_DIR", "") or "")
    except Exception:  # settings env not loaded (bare test process)
        raw = os.getenv("HUB_VISION_CROP_THUMB_DIR", "")
    return Path(raw.strip() or DEFAULT_CROP_THUMB_DIR)


def _read_thumb(path: Path) -> Optional[bytes]:
    try:
        if path.stat().st_size > MAX_THUMB_BYTES:
            return None
        return path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return None


@router.get("/api/vision/crop-thumbs/{thumb_id}")
async def get_crop_thumb(thumb_id: str) -> Response:
    """One crop thumbnail by content hash. 400 for anything that is not a
    64-char lowercase hex id; 404 when absent or pruned (kept 14 days)."""
    if not _THUMB_ID_RE.fullmatch(thumb_id or ""):
        raise HTTPException(status_code=400, detail="bad_thumb_id")
    path = _crop_thumb_dir() / f"{thumb_id}.jpg"
    data = await asyncio.to_thread(_read_thumb, path)
    if data is None:
        raise HTTPException(status_code=404, detail="thumb_not_found")
    if hashlib.sha256(data).hexdigest() != thumb_id:
        logger.error("crop_thumb_hash_mismatch id=%s", thumb_id[:12])
        raise HTTPException(status_code=404, detail="thumb_not_found")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400", "X-Content-Type-Options": "nosniff"},
    )
