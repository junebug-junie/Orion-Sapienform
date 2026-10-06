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

**Memory confirmation cards** (``source_kind`` in ``RESOLVABLE_KINDS``; memory-episode spec
sections 3 and 5, 2026-10-06). These are not closed by ``/answer`` or ``/dismiss`` (409
``ask_needs_resolution``): ``POST /api/asks/{id}/resolve`` takes Confirm / Revise (note required)
/ Reject and, in ONE transaction, closes the card and inserts the ``attention_loop_outcome`` row,
the single resolution record. After the commit it publishes ``AttentionLoopOutcomeV1`` on
``orion:attention:loop_outcome`` (consumed by orion-memory-consolidation, which also catches up
from the table, so a failed publish delays the memory update but never loses it) and the usual
``OrionAskAnsweredV1``. ``GET /api/asks`` adds ``memory_statement`` to these cards so Revise can
start from the current wording. Gated on ``MEMORY_CONFIRMATION_LOOP_ENABLED``.

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
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.memory.episode.confirmation import (
    MAX_NOTE_CHARS,
    RESOLUTION_ASK_STATUS,
    RESOLUTION_VERDICT,
    RESOLVABLE_KINDS,
    memory_id_from_loop,
    outcome_features,
    outcome_id_for,
)
from orion.schemas.ask import OrionAskAnsweredV1
from orion.schemas.attention_salience import AttentionLoopOutcomeV1

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
CHANNEL_LOOP_OUTCOME = "orion:attention:loop_outcome"
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


class AskResolveBody(BaseModel):
    resolution: Literal["confirmed", "revised", "rejected"]
    note: str = Field(default="", max_length=MAX_NOTE_CHARS)


def _confirmation_loop_enabled() -> bool:
    try:
        from scripts.settings import settings

        return bool(getattr(settings, "MEMORY_CONFIRMATION_LOOP_ENABLED", True))
    except Exception:  # settings env not loaded (bare test process)
        return os.getenv("MEMORY_CONFIRMATION_LOOP_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


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
            asks = [_row_dict(r) for r in rows]
            await _attach_memory_statements(conn, asks)
    except Exception as exc:
        _raise_store_http(exc)
    return {"ok": True, "status": status, "asks": asks}


async def _attach_memory_statements(conn: Any, asks: list[dict[str, Any]]) -> None:
    """Add ``memory_statement`` (the memory's current wording) to memory confirmation cards, so
    Revise can start from it. Fail-open: a missing shadow table must not take down the panel."""
    ids = [m for m in (memory_id_from_loop(a.get("source_ref") or "") for a in asks
                       if a.get("source_kind") in RESOLVABLE_KINDS) if m]
    if not ids:
        return
    try:
        rows = await conn.fetch(
            "SELECT memory_id::text AS memory_id, statement FROM episode_memory WHERE memory_id = ANY($1::uuid[])",
            ids,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ask_memory_statement_lookup_failed error=%s", exc)
        return
    by_id = {r["memory_id"]: r["statement"] for r in rows}
    for a in asks:
        mid = memory_id_from_loop(a.get("source_ref") or "")
        if mid and mid in by_id:
            a["memory_statement"] = by_id[mid]


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
                  AND source_kind <> ALL($4::text[])
                RETURNING ask_id, status, answer, answered_at, source_kind, source_ref
                """,
                ask_id,
                new_status,
                answer,
                list(RESOLVABLE_KINDS),
            )
            if row is None:
                existing = await conn.fetchrow(
                    "SELECT status, expires_at, source_kind FROM orion_ask WHERE ask_id = $1",
                    ask_id,
                )
    except Exception as exc:
        _raise_store_http(exc)

    if row is None:
        if existing is None:
            raise HTTPException(status_code=404, detail="ask_not_found")
        if existing["source_kind"] in RESOLVABLE_KINDS:
            # Closing these without an outcome row would orphan the memory (spec Stage 3 check 6).
            raise HTTPException(status_code=409, detail="ask_needs_resolution")
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


_INSERT_OUTCOME = """
INSERT INTO attention_loop_outcome
    (outcome_id, loop_id, theme_key, verdict, actor, note, salience_at_close, weights_version,
     features_at_close, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
ON CONFLICT (outcome_id) DO NOTHING
"""


async def _publish_loop_outcome(outcome: AttentionLoopOutcomeV1) -> bool:
    from .bus_publish import build_loop_outcome_envelope
    from .main import bus

    if bus is None or not getattr(bus, "enabled", True):
        logger.warning("ask_loop_outcome_publish_skipped outcome_id=%s reason=bus_unavailable", outcome.outcome_id)
        return False
    try:
        await bus.publish(CHANNEL_LOOP_OUTCOME, build_loop_outcome_envelope(outcome))
        return True
    except Exception as exc:
        logger.warning("ask_loop_outcome_publish_failed outcome_id=%s error=%s", outcome.outcome_id, exc)
        return False


@router.post("/api/asks/{ask_id}/resolve")
async def resolve_ask(request: Request, ask_id: str, body: AskResolveBody) -> dict[str, Any]:
    """Confirm / Revise / Reject a memory confirmation card: card + outcome in one transaction."""
    if not _confirmation_loop_enabled():
        raise HTTPException(status_code=404, detail="memory_confirmation_disabled")
    note = " ".join(body.note.split())
    if body.resolution == "revised" and not note:
        raise HTTPException(status_code=422, detail="revised_needs_note")
    new_status = RESOLUTION_ASK_STATUS[body.resolution]
    answer = body.resolution if not note else f"{body.resolution}: {note}"
    pool = _pool(request)
    outcome: Optional[AttentionLoopOutcomeV1] = None
    row: Any = None
    existing: Any = None
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE orion_ask
                    SET status = $2, answer = $3, answered_at = now()
                    WHERE ask_id = $1
                      AND status = 'open'
                      AND (expires_at IS NULL OR expires_at > now())
                      AND source_kind = ANY($4::text[])
                    RETURNING ask_id, status, answer, answered_at, source_kind, source_ref
                    """,
                    ask_id,
                    new_status,
                    answer,
                    list(RESOLVABLE_KINDS),
                )
                if row is None:
                    existing = await conn.fetchrow(
                        "SELECT status, expires_at, source_kind FROM orion_ask WHERE ask_id = $1",
                        ask_id,
                    )
                else:
                    loop_id = row["source_ref"]
                    outcome = AttentionLoopOutcomeV1(
                        outcome_id=outcome_id_for(row["ask_id"]),
                        loop_id=loop_id,
                        theme_key=loop_id,
                        verdict=RESOLUTION_VERDICT[body.resolution],  # type: ignore[arg-type]
                        actor="juniper",
                        note=note,
                        features_at_close=outcome_features(
                            resolution=body.resolution, ask_id=row["ask_id"], memory_id=memory_id_from_loop(loop_id)
                        ),
                    )
                    await conn.execute(
                        _INSERT_OUTCOME,
                        outcome.outcome_id,
                        outcome.loop_id,
                        outcome.theme_key,
                        outcome.verdict,
                        outcome.actor,
                        outcome.note,
                        float(outcome.salience_at_close),
                        outcome.weights_version,
                        json.dumps(outcome.features_at_close),
                        outcome.created_at,
                    )
    except Exception as exc:
        _raise_store_http(exc)

    if row is None or outcome is None:
        if existing is None:
            raise HTTPException(status_code=404, detail="ask_not_found")
        if existing["source_kind"] not in RESOLVABLE_KINDS:
            raise HTTPException(status_code=409, detail="ask_not_resolvable")
        current = existing["status"]
        raise HTTPException(status_code=409, detail="ask_not_open:expired" if current == "open" else f"ask_not_open:{current}")

    published_outcome = await _publish_loop_outcome(outcome)
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
        "ask_resolved ask_id=%s resolution=%s outcome_id=%s loop_id=%s published_outcome=%s",
        event.ask_id, body.resolution, outcome.outcome_id, outcome.loop_id, published_outcome,
    )
    return {
        "ok": True,
        "ask": event.model_dump(mode="json"),
        "outcome_id": outcome.outcome_id,
        "resolution": body.resolution,
        "published": published,
        "published_outcome": published_outcome,
    }


def _crop_thumb_dir() -> Path:
    try:
        from scripts.settings import settings

        raw = str(getattr(settings, "HUB_VISION_CROP_THUMB_DIR", "") or "")
    except Exception:  # settings env not loaded (bare test process)
        raw = os.getenv("HUB_VISION_CROP_THUMB_DIR", "")
    return Path(raw.strip() or DEFAULT_CROP_THUMB_DIR)


def _read_thumb(path: Path) -> Optional[bytes]:
    """Regular files only, no symlink following, never blocks on a FIFO,
    size-capped. Any OS error is "not found", never a 500."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_THUMB_BYTES:
            return None
        chunks = []
        remaining = MAX_THUMB_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        return None if len(data) > MAX_THUMB_BYTES else data
    except OSError:
        return None
    finally:
        os.close(fd)


@router.get("/api/vision/crop-thumbs/{thumb_id}")
async def get_crop_thumb(thumb_id: str) -> Response:
    """One crop thumbnail by content hash. 400 for anything that is not a
    64-char lowercase hex id; 404 when absent or pruned (vision-host keeps them 10 days)."""
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
