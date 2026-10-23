"""People Orion knows: one card per person, and Juniper's corrections (2026-10-22).

Spec: docs/superpowers/specs/2026-10-22-people-one-record-design.md section 5.

Reads the person record that orion-sql-writer keeps (``person``, ``person_link``,
``person_fact``; app/person_record.py) and lets Juniper correct it:

- ``GET  /api/people``: everyone except Orion, most-seen first, with camera days,
  a picture, facts in her words, and the open question about them (if any).
- ``POST /api/people/{person_id}/facts``: add what she knows (voice ``hub_edit``).
- ``POST /api/people/{person_id}/facts/{fact_id}/remove``: a fact stops counting
  (``removed_at``; never deleted). That category is not asked again.
- ``POST /api/people/{person_id}/status``: ``not_a_person`` for names like a
  delivery company or two people named together. Orion stops asking about them
  and their open question closes; ``active`` undoes it.

Camera links are changed where they come from (renaming on a camera card), not
here: an unlink here would be undone by the next tick that reads the same name.

Same asyncpg pool as the ask card (``app.state.memory_pg_pool``); every DB call is
awaited. Text from the database is rendered by static/js/people.js with
textContent only.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from orion.person_model.facts import SOURCE_KIND_PERSON_FACT, VOICE_HUB_EDIT, category_keys
from orion.person_model.pictures import confirmed_pictures_sql, newest_distinct

from scripts import ask_routes

logger = logging.getLogger("orion-hub.people")

router = APIRouter(tags=["people"])

MAX_FACT_CHARS = ask_routes.MAX_ANSWER_CHARS
MAX_PEOPLE = 200

_LIST_SQL = """
SELECT p.person_id, p.display_name, p.role, p.status,
       count(*) FILTER (WHERE l.source_kind = 'vision_individual') AS clusters,
       count(*) FILTER (WHERE l.source_kind = 'vision_belonging') AS belongings,
       coalesce(sum(i.distinct_days) FILTER (WHERE l.source_kind = 'vision_individual'), 0) AS days,
       max(i.last_seen_at) FILTER (WHERE l.source_kind = 'vision_individual') AS last_seen_at
FROM person p
LEFT JOIN person_link l ON l.person_id = p.person_id AND l.revoked_at IS NULL
LEFT JOIN vision_individual i ON i.individual_id = l.source_ref
WHERE p.role <> 'self'
GROUP BY p.person_id
-- A name left with no link and nothing said about it (a corrected typo) is not shown.
HAVING p.role = 'interlocutor' OR count(l.link_id) > 0
    OR EXISTS (SELECT 1 FROM person_fact f WHERE f.person_id = p.person_id AND f.removed_at IS NULL)
ORDER BY (p.status = 'active') DESC, days DESC, lower(p.display_name), p.person_id
LIMIT $1
"""

# The newest picture she herself tied to each person (orion/person_model/pictures.py):
# a camera cluster can hold more than one person, so its own pictures are not used.
_THUMB_SQL = confirmed_pictures_sql("$1::text[]")

_FACTS_SQL = """
SELECT fact_id, person_id, category, value_text, voice, created_at FROM person_fact
WHERE person_id = ANY($1::text[]) AND removed_at IS NULL
ORDER BY person_id, created_at, fact_id
"""

_OPEN_ASK_SQL = """
SELECT ask_id, question, split_part(source_ref, '|', 1) AS person_id FROM orion_ask
WHERE source_kind = $2 AND status = 'open' AND (expires_at IS NULL OR expires_at > now())
  AND split_part(source_ref, '|', 1) = ANY($1::text[])
"""


class FactBody(BaseModel):
    category: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=MAX_FACT_CHARS)


class StatusBody(BaseModel):
    status: Literal["active", "not_a_person"]


def category_label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    return value


@router.get("/api/people")
async def list_people(request: Request) -> dict[str, Any]:
    pool = ask_routes._pool(request)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(_LIST_SQL, MAX_PEOPLE)
            ids = [r["person_id"] for r in rows]
            thumbs = {pid: refs[0] for pid, refs in newest_distinct(
                [(r["person_id"], r["image_ref"], r["at"]) for r in await conn.fetch(_THUMB_SQL, ids)], 1).items()}
            fact_rows = await conn.fetch(_FACTS_SQL, ids)
            asks = {r["person_id"]: r for r in await conn.fetch(_OPEN_ASK_SQL, ids, SOURCE_KIND_PERSON_FACT)}
    except Exception as exc:
        ask_routes._raise_store_http(exc)
    facts: dict[str, list] = {}
    for f in fact_rows:
        facts.setdefault(f["person_id"], []).append({
            "fact_id": f["fact_id"], "category": f["category"], "label": category_label(f["category"]),
            "value": f["value_text"], "voice": f["voice"], "created_at": _iso(f["created_at"])})
    people = []
    for r in rows:
        pid = r["person_id"]
        ask = asks.get(pid)
        people.append({
            "person_id": pid, "name": r["display_name"], "role": r["role"], "status": r["status"],
            "clusters": int(r["clusters"]), "belongings": int(r["belongings"]), "days": int(r["days"]),
            "last_seen_at": _iso(r["last_seen_at"]), "image_ref": thumbs.get(pid),
            "facts": facts.get(pid, []),
            "open_question": {"ask_id": ask["ask_id"], "question": ask["question"]} if ask else None,
        })
    categories = [{"key": k, "label": category_label(k)} for k in category_keys()]
    return {"ok": True, "people": people, "categories": categories}


async def _person_exists(conn: Any, person_id: str) -> None:
    if not await conn.fetchval("SELECT 1 FROM person WHERE person_id = $1 AND role <> 'self'", person_id):
        raise HTTPException(status_code=404, detail="person_not_found")


@router.post("/api/people/{person_id}/facts")
async def add_fact(request: Request, person_id: str, body: FactBody) -> dict[str, Any]:
    category = body.category.strip()
    if category not in category_keys():
        raise HTTPException(status_code=422, detail="unknown_category")
    value = " ".join(body.value.split())
    if not value:
        raise HTTPException(status_code=422, detail="value_empty")
    fact_id = hashlib.md5(f"{person_id}|{category}|{VOICE_HUB_EDIT}|{uuid.uuid4().hex}".encode()).hexdigest()
    pool = ask_routes._pool(request)
    try:
        async with pool.acquire() as conn:
            await _person_exists(conn, person_id)
            await conn.execute(
                "INSERT INTO person_fact (fact_id, person_id, category, value_text, voice, source_ref) "
                "VALUES ($1, $2, $3, $4, $5, 'hub')", fact_id, person_id, category, value, VOICE_HUB_EDIT)
    except HTTPException:
        raise
    except Exception as exc:
        ask_routes._raise_store_http(exc)
    logger.info("person_fact_added person_id=%s category=%s fact_id=%s", person_id, category, fact_id)
    return {"ok": True, "fact_id": fact_id}


@router.post("/api/people/{person_id}/facts/{fact_id}/remove")
async def remove_fact(request: Request, person_id: str, fact_id: str) -> dict[str, Any]:
    pool = ask_routes._pool(request)
    try:
        async with pool.acquire() as conn:
            moved = await conn.fetchval(
                "UPDATE person_fact SET removed_at = now() WHERE fact_id = $1 AND person_id = $2 "
                "AND removed_at IS NULL RETURNING fact_id", fact_id, person_id)
    except Exception as exc:
        ask_routes._raise_store_http(exc)
    if not moved:
        raise HTTPException(status_code=404, detail="fact_not_found")
    logger.info("person_fact_removed person_id=%s fact_id=%s", person_id, fact_id)
    return {"ok": True}


@router.post("/api/people/{person_id}/status")
async def set_status(request: Request, person_id: str, body: StatusBody) -> dict[str, Any]:
    pool = ask_routes._pool(request)
    closed: list = []
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await _person_exists(conn, person_id)
                await conn.execute("UPDATE person SET status = $2, updated_at = now() WHERE person_id = $1",
                                   person_id, body.status)
                if body.status == "not_a_person":
                    # Same shape as a Dismiss: the question leaves the list and is not asked again.
                    closed = list(await conn.fetch(
                        "UPDATE orion_ask SET status = 'dismissed', answered_at = now() "
                        "WHERE source_kind = $1 AND status = 'open' AND split_part(source_ref, '|', 1) = $2 "
                        "RETURNING ask_id, status, answer, answered_at, source_kind, source_ref",
                        SOURCE_KIND_PERSON_FACT, person_id))
    except HTTPException:
        raise
    except Exception as exc:
        ask_routes._raise_store_http(exc)
    for row in closed:
        await ask_routes._publish_answered(ask_routes._event_from_row(row))
    logger.info("person_status person_id=%s status=%s asks_closed=%d", person_id, body.status, len(closed))
    return {"ok": True, "status": body.status, "closed_asks": [r["ask_id"] for r in closed]}
