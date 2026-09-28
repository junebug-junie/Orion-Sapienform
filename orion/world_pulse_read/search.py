"""Semantic search over verified readings for the reading_results introspect tool.

Each verified reading is embedded once, when it lands, and upserted through
orion-vector-writer; a question embeds only the query. Chroma is an index, not
the record: every hit is re-read from world_pulse_read_seed and re-gated.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.introspect import URL_CAP, IntrospectResultV1, clip_text
from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1, VectorUpsertV1
from orion.world_pulse_read.introspect import (
    _COLUMNS,
    _OCCURRED,
    _VERIFIED_WHERE,
    _item,
    _learned,
    _source_read,
)

UPSERT_CHANNEL = "orion:vector:semantic:upsert"
UPSERT_KIND = "vector.upsert.v1"
CANDIDATES = 20
INDEX_TEXT_CHARS = 1800
INDEX_SCAN_LIMIT = 1000
HTTP_TIMEOUT_SEC = 5.0

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


class SearchUnavailableError(RuntimeError):
    """The embedder, index, or configuration could not answer; the result is unknown."""


@dataclass(frozen=True)
class ReadingSearchConfig:
    chroma_url: str
    embed_url: str
    collection: str
    min_similarity: float
    index_interval_sec: float = 300.0
    index_batch: int = 10

    @property
    def enabled(self) -> bool:
        return bool(self.chroma_url.strip() and self.embed_url.strip() and self.collection.strip())


@dataclass(frozen=True)
class IndexPass:
    indexed: int
    pending: int


@dataclass(frozen=True)
class _Collection:
    id: str
    space: str


def document_text(row: Any) -> str | None:
    learned = _learned(row, _source_read(row))
    if not learned:
        return None
    title = str(row["title"] or "").strip()
    return clip_text(f"{title}\n\n{learned}" if title else learned, INDEX_TEXT_CHARS)[0]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def similarity(distance: float, space: str) -> float:
    # Chroma's l2 is squared euclidean; for the unit vectors bge emits,
    # ||a-b||^2 = 2 - 2cos, so cos = 1 - d/2. cosine and ip report 1 - cos.
    if space == "l2":
        return 1.0 - distance / 2.0
    return 1.0 - distance


def _base(cfg: ReadingSearchConfig) -> str:
    return f"{cfg.chroma_url.rstrip('/')}/api/v1/collections"


async def _json(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
    try:
        resp = await client.request(method, url, **kwargs)
        return resp.status_code, resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchUnavailableError(f"chroma unavailable: {type(exc).__name__}") from exc


async def embed(client: httpx.AsyncClient, cfg: ReadingSearchConfig, text: str) -> tuple[list[float], str | None]:
    req = EmbeddingGenerateV1(doc_id=f"reading-search-{uuid4()}", text=text)
    try:
        resp = await client.post(cfg.embed_url, json=req.model_dump(mode="json"))
        resp.raise_for_status()
        result = EmbeddingResultV1.model_validate(resp.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchUnavailableError(f"embedder unavailable: {type(exc).__name__}") from exc
    if not result.embedding:
        raise SearchUnavailableError("embedder returned no vector")
    return result.embedding, result.embedding_model


async def _collection(client: httpx.AsyncClient, cfg: ReadingSearchConfig) -> _Collection | None:
    status, body = await _json(client, "GET", f"{_base(cfg)}/{cfg.collection}")
    if status == 200 and isinstance(body, dict) and body.get("id"):
        space = str((body.get("metadata") or {}).get("hnsw:space") or "l2")
        return _Collection(id=str(body["id"]), space=space)
    # Chroma 0.4.24 answers a missing collection with HTTP 500 + ValueError text.
    error = str(body.get("error") or "") if isinstance(body, dict) else ""
    if status == 500 and f"Collection {cfg.collection} does not exist" in error:
        return None
    raise SearchUnavailableError(f"chroma collection lookup failed: HTTP {status}")


async def nearest(
    client: httpx.AsyncClient, cfg: ReadingSearchConfig, vector: list[float], n: int = CANDIDATES,
) -> list[tuple[str, float]]:
    """(seed_id, similarity) for the n nearest indexed readings, best first."""
    coll = await _collection(client, cfg)
    if coll is None:
        raise SearchUnavailableError("reading index not built yet")
    status, body = await _json(
        client, "POST", f"{_base(cfg)}/{coll.id}/query",
        json={"query_embeddings": [vector], "n_results": n, "include": ["distances"]},
    )
    if status != 200:
        raise SearchUnavailableError(f"chroma query failed: HTTP {status}")
    try:
        ids, distances = body["ids"][0], body["distances"][0]
        if len(ids) != len(distances) or any(not isinstance(i, str) for i in ids):
            raise ValueError("ids/distances mismatch")
        scored = [(i, similarity(float(d), coll.space)) for i, d in zip(ids, distances)]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SearchUnavailableError("malformed chroma query reply") from exc
    return sorted(scored, key=lambda s: -s[1])


async def search_readings(
    conn: Any,
    cfg: ReadingSearchConfig,
    *,
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    since: datetime | None = None,
) -> IntrospectResultV1:
    as_of = datetime.now(timezone.utc)
    vector, _ = await embed(client, cfg, query)
    scored = [s for s in await nearest(client, cfg, vector) if s[1] >= cfg.min_similarity]
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


async def verified_rows(conn: Any, scan_limit: int = INDEX_SCAN_LIMIT) -> list[Any]:
    return list(await conn.fetch(_VERIFIED_ROWS_SQL, scan_limit))


async def _stored_hashes(client: httpx.AsyncClient, cfg: ReadingSearchConfig, ids: list[str]) -> dict[str, str]:
    coll = await _collection(client, cfg)
    if coll is None:
        return {}
    status, body = await _json(
        client, "POST", f"{_base(cfg)}/{coll.id}/get", json={"ids": ids, "include": ["metadatas"]},
    )
    ids = body.get("ids") if isinstance(body, dict) else None
    metas = body.get("metadatas") if isinstance(body, dict) else None
    if status != 200 or not isinstance(ids, list) or not isinstance(metas, list) or len(ids) != len(metas):
        raise SearchUnavailableError(f"chroma get failed: HTTP {status}")
    return {str(i): str((m or {}).get("content_hash") or "") for i, m in zip(ids, metas)}


async def index_missing_readings(
    rows: list[Any],
    cfg: ReadingSearchConfig,
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
        if text:
            docs.append((row, text, content_hash(text)))
    if not docs:
        return IndexPass(indexed=0, pending=0)
    stored = await _stored_hashes(client, cfg, [str(row["seed_id"]) for row, _, _ in docs])
    stale = [d for d in docs if stored.get(str(d[0]["seed_id"])) != d[2]]
    batch = cfg.index_batch if batch is None else batch
    for row, text, digest in stale[:batch]:
        vector, model = await embed(client, cfg, text)
        occurred = row["landing_at"] or row["stage2_completed_at"] or row["handoff_at"] or row["created_at"]
        payload = VectorUpsertV1(
            doc_id=str(row["seed_id"]), collection=cfg.collection, embedding=vector,
            embedding_kind="semantic", embedding_model=model, embedding_dim=len(vector), text=text,
            meta={
                "content_hash": digest,
                "url": clip_text(row["url"], URL_CAP)[0],
                "request_id": str(row["request_id"] or ""),
                "occurred_at": occurred.isoformat() if occurred else "",
            },
        )
        await bus.publish(UPSERT_CHANNEL, BaseEnvelope(
            kind=UPSERT_KIND, source=source, payload=payload.model_dump(mode="json"),
        ))
    done = min(len(stale), batch)
    return IndexPass(indexed=done, pending=len(stale) - done)
