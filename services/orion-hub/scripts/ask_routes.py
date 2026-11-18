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

**Names on visual cards** (2026-10-15). A "same person?" card may carry ``names``
(individual_id -> name: one name for both on "Same person", one per face on
"Different people"); a multi-picture "who is this?" card may carry ``pictures``
(per picture: same person / someone else, optionally named / not sure). Both are
stored in ``orion_ask.answer_detail``; sql-writer applies them on its clock.
A named pair answer also closes, in the same transaction, any open "who is
this?" card for a named individual (``answer`` = the name, ``answer_detail`` =
``{"resolved_by": <pair ask_id>}``), so Juniper is not asked again.

**"I don't know"** (2026-10-27). ``POST /api/asks/{id}/dismiss`` takes an
optional body ``{"reason": "dont_know"}``, only on a "who is this?" card
(``vision_individual``; 422 ``dont_know_only_on_who_is_this_cards`` otherwise).
It is a dismissal: ``status='dismissed'``, ``answer`` NULL, so no name is ever
stored, and ``answer_detail = {"dont_know": true}``. sql-writer treats that
like an expired card (asked again after ``VISION_ASK_COOLDOWN_DAYS``); a plain
dismissal still blocks that individual for good. Typed answers are never
filtered by a word list: the explicit button is the fix for non-answers
("no", "unsure") that used to be saved as names.

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
from typing import Annotated, Any, Literal, Optional

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
    revision_problem,
)
from orion.schemas.ask import OrionAskAnsweredV1
from orion.schemas.attention_salience import AttentionLoopOutcomeV1

try:
    from asyncpg.exceptions import (
        InterfaceError as _AsyncpgInterfaceError,
        PostgresConnectionError as _AsyncpgConnectionError,
        UndefinedColumnError as _AsyncpgUndefinedColumnError,
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
    _AsyncpgUndefinedColumnError = None  # type: ignore[misc, assignment]
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
    "ask_id, asked_of, question, evidence_refs, image_ref, image_refs, status, answer, "
    "answered_at, created_at, expires_at, source_kind, source_ref"
)


# "Are these the same person?" (orion-sql-writer opens these, 2026-10-15): the
# only answers are yes / no; "not sure" is Dismiss. sql-writer reads the
# answer as a verdict, so free text is refused here instead of being dropped there.
SAME_INDIVIDUAL_KIND = "vision_same_individual"
# "Is this <name>?" (orion-sql-writer person learning, 2026-10-20): yes / no too;
# "can't tell" is Dismiss. source_ref "<individual_id>|<name>".
PERSON_CONFIRM_KIND = "vision_person_confirm"
YES_NO_KINDS = (SAME_INDIVIDUAL_KIND, PERSON_CONFIRM_KIND)
YES_NO_ANSWERS = ("yes", "no")
WHO_IS_THIS_KIND = "vision_individual"
PAIR_SEP = "|"
# orion.substrate.adapters.vision_individual.MAX_LABEL_LENGTH: a longer name
# would be cut there, so it is refused here.
MAX_NAME_CHARS = 120
MAX_PICTURES = 4


# Explicit ::text casts: Postgres cannot infer $3's type from "$3 IS NULL"
# (AmbiguousParameterError) -- caught in review, pinned by a real-Postgres test.
CLOSE_ASK_SQL = """
UPDATE orion_ask
SET status = $2, answer = $3::text, answered_at = now(), answer_detail = $7::jsonb
WHERE ask_id = $1
  AND status = 'open'
  AND (expires_at IS NULL OR expires_at > now())
  AND source_kind <> ALL($4::text[])
  AND (source_kind <> ALL($5::text[]) OR $3::text IS NULL OR $3::text = ANY($6::text[]))
RETURNING ask_id, status, answer, answered_at, source_kind, source_ref
"""

# A named "same person?" answer closes any open "who is this?" card for a named
# individual: she already said who it is. Same transaction as the pair answer.
CLOSE_NAMED_WHO_SQL = """
UPDATE orion_ask a
SET status = 'answered', answer = n.name, answered_at = now(),
    answer_detail = jsonb_build_object('resolved_by', $2::text)
FROM jsonb_each_text($1::jsonb) AS n(individual_id, name)
WHERE a.source_kind = $3::text
  AND a.source_ref = n.individual_id
  AND a.status = 'open'
  AND (a.expires_at IS NULL OR a.expires_at > now())
RETURNING a.ask_id, a.status, a.answer, a.answered_at, a.source_kind, a.source_ref
"""


class PictureAnswer(BaseModel):
    image_ref: str = Field(min_length=1, max_length=500)
    verdict: Literal["same", "different", "unsure"]
    name: str = Field(default="", max_length=MAX_ANSWER_CHARS)


class AskDismissBody(BaseModel):
    # "dont_know": Juniper does not know who it is (who-is-this cards only).
    reason: Optional[Literal["dont_know"]] = None


class AskAnswerBody(BaseModel):
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)
    # "Same person?" cards only: individual_id -> name (empty names are dropped).
    names: dict[str, Annotated[str, Field(max_length=MAX_ANSWER_CHARS)]] = Field(default_factory=dict)
    # Multi-picture "who is this?" cards only: one entry per picture.
    pictures: list[PictureAnswer] = Field(default_factory=list, max_length=MAX_PICTURES)


def clean_name(value: Any) -> str:
    name = " ".join(str(value or "").split())
    if len(name) > MAX_NAME_CHARS:
        raise HTTPException(status_code=422, detail="name_too_long")
    return name


def pair_answer_detail(answer: str, source_ref: str, names: dict[str, str]) -> Optional[dict[str, Any]]:
    """``{"names": {id: name}}`` for a "same person?" answer, or None when no name was typed.

    "yes" (same person): one name for the pair, given to both ids. "no": one per face."""
    cleaned = {str(k): clean_name(v) for k, v in (names or {}).items()}
    cleaned = {k: v for k, v in cleaned.items() if v}
    if not cleaned:
        return None
    ids = [i for i in str(source_ref or "").split(PAIR_SEP) if i]
    if len(ids) != 2 or any(k not in ids for k in cleaned):
        raise HTTPException(status_code=422, detail="name_for_unknown_individual")
    if answer == "yes":
        distinct = set(cleaned.values())
        if len(distinct) > 1:
            raise HTTPException(status_code=422, detail="same_person_takes_one_name")
        name = distinct.pop()
        cleaned = {i: name for i in ids}
    return {"names": cleaned}


def _image_refs(row: Any) -> list[str]:
    refs = row["image_refs"]
    if isinstance(refs, str):
        try:
            refs = json.loads(refs)
        except ValueError:
            refs = []
    refs = [str(r).strip() for r in (refs or []) if isinstance(r, str) and r.strip()]
    if not refs and row["image_ref"]:
        refs = [str(row["image_ref"])]
    return refs


def picture_answer_detail(answer: str, image_refs: list[str], pictures: list[PictureAnswer]) -> Optional[dict[str, Any]]:
    """``{"pictures": [...]}`` for a "who is this?" card with more than one picture.

    The first picture is the one the question is about (faces come first): its
    name is the answer. Every other picture is "same" (gets the answer as its
    name), "different" (someone else; the name is optional) or "unsure"."""
    if not pictures:
        return None
    if len(image_refs) < 2:
        raise HTTPException(status_code=422, detail="pictures_need_a_multi_picture_card")
    index: dict[str, int] = {}
    for i, ref in enumerate(image_refs):
        index.setdefault(ref, i)  # first position wins, as the card shows it (askImageViews)
    out: dict[int, dict[str, Any]] = {0: {"image_ref": image_refs[0], "index": 0, "verdict": "same", "name": answer}}
    seen: set[str] = set()
    for p in pictures:
        if p.image_ref not in index:
            raise HTTPException(status_code=422, detail="picture_not_on_card")
        if p.image_ref in seen:
            raise HTTPException(status_code=422, detail="picture_listed_twice")
        seen.add(p.image_ref)
        i = index[p.image_ref]
        name = clean_name(p.name)
        if i == 0:
            if p.verdict != "same" or (name and name != answer):
                raise HTTPException(status_code=422, detail="first_picture_is_the_answer")
            continue
        if p.verdict == "same" and name and name != answer:
            raise HTTPException(status_code=422, detail="same_picture_takes_the_answer")
        out[i] = {"image_ref": p.image_ref, "index": i, "verdict": p.verdict,
                  "name": answer if p.verdict == "same" else (name if p.verdict == "different" and name else None)}
    return {"pictures": [out[i] for i in sorted(out)]}


def confirm_picture_detail(answer: str, image_refs: list[str], pictures: list[PictureAnswer]) -> dict[str, Any]:
    """``{"pictures": [...]}`` for an "Is this <name>?" card answered picture by
    picture (2026-10-23): "same" = it is <name>, "different" = it is not,
    "unsure". Each picture at most once (one left out is "unsure"); no names
    (the card names one person).
    The answer is "yes" exactly when at least one picture is <name>."""
    if len(image_refs) < 2:
        raise HTTPException(status_code=422, detail="pictures_need_a_multi_picture_card")
    index: dict[str, int] = {}
    for i, ref in enumerate(image_refs):
        index.setdefault(ref, i)
    out: dict[int, dict[str, Any]] = {}
    for p in pictures:
        if p.image_ref not in index:
            raise HTTPException(status_code=422, detail="picture_not_on_card")
        i = index[p.image_ref]
        if i in out:
            raise HTTPException(status_code=422, detail="picture_listed_twice")
        if clean_name(p.name):
            raise HTTPException(status_code=422, detail="no_names_on_is_this_cards")
        out[i] = {"image_ref": p.image_ref, "index": i, "verdict": p.verdict}
    # A picture the card could not show gets no answer: it counts as "unsure".
    for ref, i in index.items():
        out.setdefault(i, {"image_ref": ref, "index": i, "verdict": "unsure"})
    if (answer == "yes") != any(v["verdict"] == "same" for v in out.values()):
        raise HTTPException(status_code=422, detail="answer_does_not_match_pictures")
    return {"pictures": [out[i] for i in sorted(out)]}


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
    # A missing column is a migration not applied yet (e.g. orion_ask.answer_detail): same 503.
    schema_errors = tuple(e for e in (_AsyncpgUndefinedTableError, _AsyncpgUndefinedColumnError) if e is not None)
    if schema_errors and isinstance(exc, schema_errors):
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
    imgs = out.get("image_refs")
    if isinstance(imgs, str):
        try:
            imgs = json.loads(imgs)
        except ValueError:
            imgs = []
    imgs = [str(i) for i in (imgs or []) if isinstance(i, str) and i.strip()]
    # Rows from before image_refs (2026-10-15) carry only image_ref.
    if not imgs and out.get("image_ref"):
        imgs = [str(out["image_ref"])]
    out["image_refs"] = imgs
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
    payload = event.model_dump(mode="json")
    # Additive field: left out when empty, so a consumer built before it
    # (extra="forbid") still accepts every answer that carries no names.
    if not payload.get("names"):
        payload.pop("names", None)
    return BaseEnvelope(
        kind=ASK_ANSWERED_KIND,
        source=_source_ref(),
        payload=payload,
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


def _event_from_row(row: Any, names: Optional[dict[str, str]] = None) -> OrionAskAnsweredV1:
    answered_at = row["answered_at"]
    if isinstance(answered_at, datetime) and answered_at.tzinfo is None:
        answered_at = answered_at.replace(tzinfo=timezone.utc)
    return OrionAskAnsweredV1(
        ask_id=row["ask_id"],
        status=row["status"],
        answer=row["answer"],
        answered_at=answered_at or datetime.now(timezone.utc),
        source_kind=row["source_kind"],
        source_ref=row["source_ref"],
        names=dict(names or {}),
    )


async def _answer_detail(conn: Any, ask_id: str, answer: Optional[str], names: dict[str, str],
                         pictures: list[PictureAnswer], dont_know: bool = False) -> Optional[dict[str, Any]]:
    """Validate typed names against the card (locks the row) and build ``answer_detail``.
    Raises 422 for names on a card that cannot take them, or "I don't know" on a
    card that is not "who is this?"; None when nothing was typed."""
    if not names and not pictures and not dont_know:
        return None
    card = await conn.fetchrow(
        "SELECT source_kind, source_ref, image_ref, image_refs, status, expires_at FROM orion_ask "
        "WHERE ask_id = $1 FOR UPDATE",
        ask_id,
    )
    expires = card["expires_at"] if card is not None else None
    if isinstance(expires, datetime) and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if card is None or card["status"] != "open" or (expires is not None and expires <= datetime.now(timezone.utc)):
        return None  # CLOSE_ASK_SQL moves nothing either: the usual 404 / 409 below
    detail: dict[str, Any] = {}
    if dont_know:
        if card["source_kind"] != WHO_IS_THIS_KIND:
            raise HTTPException(status_code=422, detail="dont_know_only_on_who_is_this_cards")
        detail["dont_know"] = True
    if names:
        if card["source_kind"] != SAME_INDIVIDUAL_KIND:
            raise HTTPException(status_code=422, detail="names_only_on_same_person_cards")
        if answer in YES_NO_ANSWERS:  # anything else is refused by CLOSE_ASK_SQL (422 below)
            detail.update(pair_answer_detail(answer, card["source_ref"], names) or {})
    if pictures:
        if card["source_kind"] == PERSON_CONFIRM_KIND:
            if answer in YES_NO_ANSWERS:  # anything else is refused by CLOSE_ASK_SQL (422 below)
                detail.update(confirm_picture_detail(answer, _image_refs(card), pictures))
        elif card["source_kind"] == WHO_IS_THIS_KIND:
            detail.update(picture_answer_detail(answer or "", _image_refs(card), pictures) or {})
        else:
            raise HTTPException(status_code=422, detail="pictures_only_on_who_is_this_cards")
    return detail or None


async def _close_ask(
    request: Request,
    ask_id: str,
    *,
    new_status: Literal["answered", "dismissed"],
    answer: Optional[str],
    names: Optional[dict[str, str]] = None,
    pictures: Optional[list[PictureAnswer]] = None,
    dont_know: bool = False,
) -> dict[str, Any]:
    pool = _pool(request)
    row: Any = None
    existing: Any = None
    detail: Optional[dict[str, Any]] = None
    closed: list[Any] = []
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                detail = await _answer_detail(conn, ask_id, answer, names or {}, pictures or [], dont_know)
                row = await conn.fetchrow(
                    CLOSE_ASK_SQL,
                    ask_id,
                    new_status,
                    answer,
                    list(RESOLVABLE_KINDS),
                    list(YES_NO_KINDS),
                    list(YES_NO_ANSWERS),
                    None if detail is None else json.dumps(detail),
                )
                if row is None:
                    existing = await conn.fetchrow(
                        "SELECT status, expires_at, source_kind FROM orion_ask WHERE ask_id = $1",
                        ask_id,
                    )
                elif detail and detail.get("names"):
                    closed = list(await conn.fetch(
                        CLOSE_NAMED_WHO_SQL, json.dumps(detail["names"]), row["ask_id"], WHO_IS_THIS_KIND))
    except Exception as exc:
        _raise_store_http(exc)

    if row is None:
        if existing is None:
            raise HTTPException(status_code=404, detail="ask_not_found")
        if (existing["source_kind"] in YES_NO_KINDS and new_status == "answered"
                and answer not in YES_NO_ANSWERS and existing["status"] == "open"
                and (existing["expires_at"] is None or existing["expires_at"] > datetime.now(timezone.utc))):
            raise HTTPException(status_code=422, detail="answer_must_be_yes_or_no")
        if existing["source_kind"] in RESOLVABLE_KINDS:
            # Closing these without an outcome row would orphan the memory (spec Stage 3 check 6).
            raise HTTPException(status_code=409, detail="ask_needs_resolution")
        current = existing["status"]
        detail_text = f"ask_not_open:{current}"
        if current == "open":
            detail_text = "ask_not_open:expired"
        raise HTTPException(status_code=409, detail=detail_text)

    event = _event_from_row(row, (detail or {}).get("names"))
    published = await _publish_answered(event)
    logger.info(
        "ask_closed ask_id=%s status=%s source_kind=%s source_ref=%s names=%d pictures=%d dont_know=%s "
        "published=%s",
        event.ask_id,
        event.status,
        event.source_kind,
        event.source_ref,
        len(event.names),
        len((detail or {}).get("pictures") or []),
        bool((detail or {}).get("dont_know")),
        published,
    )
    closed_ids = []
    for c in closed:
        closed_event = _event_from_row(c)
        closed_published = await _publish_answered(closed_event)
        closed_ids.append(closed_event.ask_id)
        logger.info("ask_closed_by_pair ask_id=%s pair_ask_id=%s individual_id=%s published=%s",
                    closed_event.ask_id, event.ask_id, closed_event.source_ref, closed_published)
    return {"ok": True, "ask": event.model_dump(mode="json"), "published": published, "closed_asks": closed_ids}


@router.post("/api/asks/{ask_id}/answer")
async def answer_ask(request: Request, ask_id: str, body: AskAnswerBody) -> dict[str, Any]:
    answer = " ".join(body.answer.split())
    if answer.lower() in YES_NO_ANSWERS:
        answer = answer.lower()
    if not answer:
        raise HTTPException(status_code=422, detail="answer_empty")
    if len(body.names) > 2:
        raise HTTPException(status_code=422, detail="name_for_unknown_individual")
    return await _close_ask(request, ask_id, new_status="answered", answer=answer,
                            names=body.names, pictures=body.pictures)


@router.post("/api/asks/{ask_id}/dismiss")
async def dismiss_ask(request: Request, ask_id: str, body: Optional[AskDismissBody] = None) -> dict[str, Any]:
    dont_know = body is not None and body.reason == "dont_know"
    return await _close_ask(request, ask_id, new_status="dismissed", answer=None, dont_know=dont_know)


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
    if body.resolution == "revised":
        # Structural checks first (no DB): empty or too short to be wording. "Unchanged" needs the
        # memory's current statement and is checked inside the transaction below.
        problem = revision_problem(note, None)
        if problem:
            raise HTTPException(status_code=422, detail=problem)
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
                    if body.resolution == "revised":
                        current = await conn.fetchval(
                            "SELECT statement FROM episode_memory WHERE memory_id = $1::uuid",
                            memory_id_from_loop(loop_id),
                        )
                        problem = revision_problem(note, current)
                        if problem:  # raising inside the transaction rolls the card update back
                            raise HTTPException(status_code=422, detail=problem)
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
    64-char lowercase hex id; 404 when absent (vision-host keeps them forever by default
    since 2026-10-20; a positive VISION_CROP_THUMB_RETENTION_DAYS prunes them)."""
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
