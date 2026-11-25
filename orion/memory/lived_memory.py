"""Lived memory: the three stores the chat recall path reads by meaning (issue #3533).

- ``episode_memory``: Orion's first-person distilled memories.
- ``memory_crystallizations`` kind ``stance``: Juniper's heaviest messages, verbatim.
- ``person_fact``: Juniper's own words from the People page.

This module owns the SQL that says which rows may be served and how each row becomes one
indexable document. orion-recall indexes the documents into Chroma and serves them in chat;
the mesh-edge check (``orion/memory/mesh_edge.py``) reads the same rows so a line Orion can
see in chat cannot be pasted into public output.

Serving rules (the reader re-applies them at query time; Chroma is an index, never the record):

- an ``episode_memory`` row is served only when ``status = 'active'``, it is not rejected, it
  has not expired, and no live row names it in ``supersedes_memory_id`` (a correction
  replaces what it corrects);
- a stance crystallization is served only when ``status = 'active'``;
- a person fact is served only when ``removed_at IS NULL`` and its person is active.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

KIND_EPISODE = "episode_memory"
KIND_STANCE = "stance"
KIND_PERSON_FACT = "person_fact"
KINDS = (KIND_EPISODE, KIND_STANCE, KIND_PERSON_FACT)

COLLECTION = "orion_lived_memory"

# Rows that may be served. Each query returns the same column shape:
# record_id, purpose, text, occurred_at, importance, is_correction, confirmed, label, created_at.
EPISODE_SQL = """
SELECT m.memory_id::text AS record_id,
       m.purpose AS purpose,
       m.statement AS text,
       coalesce(m.occurred_at, m.created_at) AS occurred_at,
       m.strength::float8 AS importance,
       (m.supersedes_memory_id IS NOT NULL) AS is_correction,
       (m.confirmation_state = 'confirmed') AS confirmed,
       NULL::text AS label,
       m.created_at AS created_at
FROM episode_memory m
WHERE m.status = 'active'
  AND m.confirmation_state IS DISTINCT FROM 'rejected'
  AND (m.expires_at IS NULL OR m.expires_at > now())
  AND length(trim(m.statement)) > 0
  AND NOT EXISTS (
      SELECT 1 FROM episode_memory c
      WHERE c.supersedes_memory_id = m.memory_id AND c.status = 'active'
  )
"""

# "What was true then" (issue #3659, bi-temporal validity). EPISODE_SQL above serves only what is
# currently true. This reader also serves a row an UPDATE retired, when it was still true at $1
# (asyncpg timestamptz): it happened by then, and its invalid_at is after then. A CORRECTION's
# old row is never served (it was never true), and neither is anything rejected.
EPISODE_VALID_AT_SQL = """
SELECT m.memory_id::text AS record_id,
       m.purpose AS purpose,
       m.statement AS text,
       coalesce(m.occurred_at, m.created_at) AS occurred_at,
       m.strength::float8 AS importance,
       (m.supersedes_memory_id IS NOT NULL) AS is_correction,
       (m.confirmation_state = 'confirmed') AS confirmed,
       NULL::text AS label,
       m.created_at AS created_at,
       m.invalid_at AS invalid_at
FROM episode_memory m
WHERE m.confirmation_state IS DISTINCT FROM 'rejected'
  AND length(trim(m.statement)) > 0
  AND coalesce(m.occurred_at, m.created_at) <= $1
  AND (m.expires_at IS NULL OR m.expires_at > $1)
  AND (
      (m.status = 'active'
       AND NOT EXISTS (SELECT 1 FROM episode_memory c
                       WHERE c.supersedes_memory_id = m.memory_id AND c.status = 'active'))
      OR (m.status = 'superseded' AND m.invalid_reason = 'update' AND m.invalid_at > $1)
  )
"""


async def fetch_valid_at(conn: Any, as_of: datetime) -> list[LivedRecord]:
    """Episode memories true at ``as_of`` (asyncpg connection or pool): the past-tense read."""
    rows = await conn.fetch(EPISODE_VALID_AT_SQL, as_of)
    return records_from_rows(KIND_EPISODE, [dict(r) for r in rows])


# "What was true at some point in [$1, $2)" (issue #3674): the as-of read over a window instead of
# one instant, for a past-tense chat question ("where was I in early October?"). Same gate as
# EPISODE_VALID_AT_SQL: it had happened by the window's end, and an UPDATE retired it, if at all,
# after the window began. A CORRECTION's old row is still never served.
EPISODE_VALID_DURING_SQL = """
SELECT m.memory_id::text AS record_id,
       m.purpose AS purpose,
       m.statement AS text,
       coalesce(m.occurred_at, m.created_at) AS occurred_at,
       m.strength::float8 AS importance,
       (m.supersedes_memory_id IS NOT NULL) AS is_correction,
       (m.confirmation_state = 'confirmed') AS confirmed,
       NULL::text AS label,
       m.created_at AS created_at,
       m.invalid_at AS invalid_at
FROM episode_memory m
WHERE m.confirmation_state IS DISTINCT FROM 'rejected'
  AND length(trim(m.statement)) > 0
  AND coalesce(m.occurred_at, m.created_at) < $2
  AND (m.expires_at IS NULL OR m.expires_at > $1)
  AND (
      (m.status = 'active'
       AND NOT EXISTS (SELECT 1 FROM episode_memory c
                       WHERE c.supersedes_memory_id = m.memory_id AND c.status = 'active'))
      OR (m.status = 'superseded' AND m.invalid_reason = 'update' AND m.invalid_at > $1)
  )
"""


async def fetch_valid_during(conn: Any, start: datetime, end: datetime, *,
                             ids: Iterable[str] | None = None, retired_only: bool = False) -> list[LivedRecord]:
    """Episode memories true at some point in ``[start, end)`` (asyncpg connection or pool).

    ``retired_only`` keeps only the rows an update has since retired (``invalid_at`` set): the ones
    the currently-true reader no longer serves. ``ids`` limits the read to those memory ids (the
    re-check before render)."""
    sql = EPISODE_VALID_DURING_SQL
    args: list[Any] = [start, end]
    if retired_only:
        sql += " AND m.invalid_at IS NOT NULL"
    if ids is not None:
        sql += " AND m.memory_id::text = ANY($3::text[])"
        args.append(list(ids))
    rows = await conn.fetch(sql, *args)
    return records_from_rows(KIND_EPISODE, [dict(r) for r in rows])


STANCE_SQL = """
SELECT c.crystallization_id::text AS record_id,
       'stance' AS purpose,
       c.summary AS text,
       c.created_at AS occurred_at,
       c.salience::float8 AS importance,
       false AS is_correction,
       false AS confirmed,
       NULL::text AS label,
       c.created_at AS created_at
FROM memory_crystallizations c
WHERE c.kind = 'stance' AND c.status = 'active' AND length(trim(c.summary)) > 0
"""

PERSON_FACT_SQL = """
SELECT f.fact_id AS record_id,
       f.category AS purpose,
       f.value_text AS text,
       f.created_at AS occurred_at,
       0.9::float8 AS importance,
       false AS is_correction,
       true AS confirmed,
       p.display_name AS label,
       f.created_at AS created_at
FROM person_fact f
JOIN person p ON p.person_id = f.person_id
WHERE f.removed_at IS NULL AND p.status = 'active' AND p.merged_into IS NULL
  AND length(trim(f.value_text)) > 0
"""

SQL_BY_KIND: dict[str, str] = {
    KIND_EPISODE: EPISODE_SQL,
    KIND_STANCE: STANCE_SQL,
    KIND_PERSON_FACT: PERSON_FACT_SQL,
}

# Id filters for the final re-check of chosen rows (asyncpg $1 = text[]).
ID_FILTER: dict[str, str] = {
    KIND_EPISODE: " AND m.memory_id::text = ANY($1::text[])",
    KIND_STANCE: " AND c.crystallization_id::text = ANY($1::text[])",
    KIND_PERSON_FACT: " AND f.fact_id = ANY($1::text[])",
}

MAX_DOC_CHARS = 1200

# Link-following (issue #3534): which memories share a referent, and which names point at one.
# Only memories that are servable right now become candidates (the reader intersects with its
# live snapshot), so these two reads need no serving gate of their own.
REFERENT_EDGES_SQL = """
SELECT r.memory_id::text AS memory_id, r.referent_key AS referent_key
FROM episode_memory_referent r
"""

# Name-class aliases only, mapped to the referent key the edges use. Descriptor aliases
# ("that", "home", "the mesh") are generic words and are never read.
REFERENT_NAMES_SQL = """
SELECT a.alias_norm AS alias, k.alias_norm AS referent_key
FROM referent_alias a
JOIN referent_alias k ON k.node_id = a.node_id AND k.alias_class = 'key'
WHERE a.alias_class = 'name'
"""


@dataclass(frozen=True)
class LivedRecord:
    kind: str
    record_id: str
    purpose: str
    text: str            # what is embedded, searched, reranked and shown
    occurred_at: datetime | None
    importance: float    # [0, 1]
    is_correction: bool = False
    confirmed: bool = False
    created_at: datetime | None = None  # when Orion stored it (replay evals filter on this)
    # Set only by the as-of readers: when an update retired this row (true then, not now).
    invalid_at: datetime | None = None

    @property
    def doc_id(self) -> str:
        return doc_id(self.kind, self.record_id)

    @property
    def date(self) -> str:
        return self.occurred_at.date().isoformat() if self.occurred_at else "undated"


def doc_id(kind: str, record_id: str) -> str:
    return f"{kind}:{record_id}"


def split_doc_id(value: str) -> tuple[str, str] | None:
    kind, sep, rid = str(value or "").partition(":")
    if not sep or kind not in KINDS or not rid:
        return None
    return kind, rid


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _clamp01(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def record_from_row(kind: str, row: Mapping[str, Any]) -> LivedRecord | None:
    text = " ".join(str(row.get("text") or "").split())
    if not text:
        return None
    if kind == KIND_PERSON_FACT:
        # A bare fact value ("she/her", "7") means nothing without who and what it is about.
        name = " ".join(str(row.get("label") or "").split()) or "someone"
        category = str(row.get("purpose") or "fact").replace("_", " ")
        text = f"{name}, {category}: {text}"
    return LivedRecord(
        kind=kind,
        record_id=str(row.get("record_id") or ""),
        purpose=str(row.get("purpose") or kind),
        text=text[:MAX_DOC_CHARS],
        occurred_at=_aware(row.get("occurred_at")),
        importance=_clamp01(row.get("importance"), 0.5),
        is_correction=bool(row.get("is_correction")),
        confirmed=bool(row.get("confirmed")),
        created_at=_aware(row.get("created_at")),
        invalid_at=_aware(row.get("invalid_at")),
    )


def records_from_rows(kind: str, rows: Iterable[Mapping[str, Any]]) -> list[LivedRecord]:
    out = []
    for row in rows:
        rec = record_from_row(kind, row)
        if rec is not None and rec.record_id:
            out.append(rec)
    return out


def index_doc(rec: LivedRecord) -> tuple[str, str, dict[str, Any]]:
    """(doc_id, text, meta) for ``orion.introspect.semantic_index.index_docs``. Chroma meta values
    must be scalars."""
    meta: dict[str, Any] = {"kind": rec.kind, "record_id": rec.record_id, "purpose": rec.purpose}
    if rec.occurred_at is not None:
        meta["occurred_ts"] = rec.occurred_at.timestamp()
    return rec.doc_id, rec.text, meta


async def fetch_live_records(conn: Any, *, kinds: Iterable[str] = KINDS,
                             ids_by_kind: Mapping[str, list[str]] | None = None) -> list[LivedRecord]:
    """Every servable record (asyncpg connection or pool). With ``ids_by_kind`` only those ids,
    still gated: a row superseded or retired since it was indexed comes back missing."""
    out: list[LivedRecord] = []
    for kind in kinds:
        sql = SQL_BY_KIND[kind]
        if ids_by_kind is not None:
            ids = list(ids_by_kind.get(kind) or [])
            if not ids:
                continue
            rows = await conn.fetch(sql + ID_FILTER[kind], ids)
        else:
            rows = await conn.fetch(sql)
        out.extend(records_from_rows(kind, [dict(r) for r in rows]))
    return out
