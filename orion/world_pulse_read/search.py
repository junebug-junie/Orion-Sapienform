"""Semantic search over verified readings for the reading_results introspect tool.

Each verified reading is embedded once, when it lands, and upserted through
orion-vector-writer; a question embeds only the query. Chroma is an index, not
the record: every hit is re-read from world_pulse_read_seed and re-gated.
Shared plumbing: orion/introspect/semantic_index.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from orion.introspect.semantic_index import (
    HTTP_TIMEOUT_SEC,
    UPSERT_CHANNEL,
    UPSERT_KIND,
    IndexPass,
    SearchConfig,
    SearchUnavailableError,
    content_hash,
    index_docs,
    similarity,
)
from orion.introspect.semantic_index import embed as _embed
from orion.introspect.semantic_index import nearest as _nearest
from orion.schemas.introspect import URL_CAP, IntrospectResultV1, clip_text
from orion.world_pulse_read.introspect import (
    _COLUMNS,
    _OCCURRED,
    _VERIFIED_WHERE,
    _item,
    _learned,
    _source_read,
)

__all__ = [
    "CANDIDATES", "HTTP_TIMEOUT_SEC", "INDEX_SCAN_LIMIT", "INDEX_TEXT_CHARS", "UPSERT_CHANNEL",
    "UPSERT_KIND", "IndexPass", "ReadingSearchConfig", "SearchUnavailableError", "content_hash",
    "document_text", "embed", "gated_results", "index_missing_readings", "nearest",
    "rank_readings", "search_readings", "similarity", "verified_rows",
]

CANDIDATES = 20
INDEX_TEXT_CHARS = 1800
INDEX_SCAN_LIMIT = 1000
_DOC_PREFIX = "reading-search"

ReadingSearchConfig = SearchConfig

_VERIFIED_ROWS_SQL = f"""
SELECT {_COLUMNS} FROM world_pulse_read_seed s
WHERE {_VERIFIED_WHERE}
ORDER BY {_OCCURRED} DESC, s.seed_id DESC
LIMIT $1
"""

_ROWS_BY_ID_SQL = f"""
SELECT {_COLUMNS} FROM world_pulse_read_seed s
WHERE s.seed_id = ANY($1::text[])
  AND {_VERIFIED_WHERE}
  AND ($2::timestamptz IS NULL OR {_OCCURRED} >= $2)
"""


def document_text(row: Any) -> str | None:
    learned = _learned(row, _source_read(row))
    if not learned:
        return None
    title = str(row["title"] or "").strip()
    return clip_text(f"{title}\n\n{learned}" if title else learned, INDEX_TEXT_CHARS)[0]


async def embed(client: httpx.AsyncClient, cfg: SearchConfig, text: str) -> tuple[list[float], str | None]:
    return await _embed(client, cfg, text, doc_prefix=_DOC_PREFIX)


async def nearest(
    client: httpx.AsyncClient, cfg: SearchConfig, vector: list[float], n: int = CANDIDATES,
) -> list[tuple[str, float]]:
    """(seed_id, similarity) for the n nearest indexed readings, best first."""
    return await _nearest(client, cfg, vector, n)


async def rank_readings(
    client: httpx.AsyncClient, cfg: SearchConfig, query: str,
) -> list[tuple[str, float]]:
    """Embed the query once; (seed_id, similarity) hits at or above the floor, best first."""
    vector, _ = await embed(client, cfg, query)
    return [s for s in await nearest(client, cfg, vector) if s[1] >= cfg.min_similarity]


async def gated_results(
    conn: Any,
    scored: list[tuple[str, float]],
    *,
    limit: int,
    since: datetime | None = None,
) -> IntrospectResultV1:
    """Re-read ranked hits from Postgres; only verified readings survive."""
    as_of = datetime.now(timezone.utc)
    if not scored:
        return IntrospectResultV1(ok=True, operation="reading_result", as_of=as_of, total_available=0)
    rows = {str(r["seed_id"]): r for r in await conn.fetch(_ROWS_BY_ID_SQL, [sid for sid, _ in scored], since)}
    hits = [
        (rows[sid], sim) for sid, sim in scored
        if sid in rows and _learned(rows[sid], _source_read(rows[sid]))
    ]
    items = []
    for row, sim in hits[:limit]:
        item = _item(row, request_id=str(row["request_id"]) if row["request_id"] else None)
        items.append(item.model_copy(update={"extra": {**item.extra, "similarity": round(sim, 3)}}))
    return IntrospectResultV1(
        ok=True, operation="reading_result", as_of=as_of, total_available=len(hits), items=items,
    )


async def search_readings(
    conn: Any,
    cfg: SearchConfig,
    *,
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    since: datetime | None = None,
) -> IntrospectResultV1:
    return await gated_results(conn, await rank_readings(client, cfg, query), limit=limit, since=since)


async def verified_rows(conn: Any, scan_limit: int = INDEX_SCAN_LIMIT) -> list[Any]:
    return list(await conn.fetch(_VERIFIED_ROWS_SQL, scan_limit))


async def index_missing_readings(
    rows: list[Any],
    cfg: SearchConfig,
    *,
    client: httpx.AsyncClient,
    bus: Any,
    source: Any,
    batch: int | None = None,
) -> IndexPass:
    """Embed and upsert verified readings the index lacks or holds stale text for."""
    docs = []
    for row in rows:
        text = document_text(row)
        if not text:
            continue
        occurred = row["landing_at"] or row["stage2_completed_at"] or row["handoff_at"] or row["created_at"]
        docs.append((str(row["seed_id"]), text, {
            "url": clip_text(row["url"], URL_CAP)[0],
            "request_id": str(row["request_id"] or ""),
            "occurred_at": occurred.isoformat() if occurred else "",
        }))
    return await index_docs(
        docs, cfg, client=client, bus=bus, source=source, doc_prefix=_DOC_PREFIX, batch=batch,
    )
