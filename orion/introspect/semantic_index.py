"""Shared search-by-meaning plumbing for orion-introspect responders.

A record is embedded once (vector-host HTTP /embedding, which persists nothing)
and upserted through orion-vector-writer into a per-domain Chroma collection;
a question embeds only the query. Chroma is an index, never the record: every
caller re-reads each hit from its own tables and re-gates it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import uuid4

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1, VectorUpsertV1

UPSERT_CHANNEL = "orion:vector:semantic:upsert"
UPSERT_KIND = "vector.upsert.v1"
HTTP_TIMEOUT_SEC = 5.0


class SearchUnavailableError(RuntimeError):
    """The embedder, index, or configuration could not answer; the result is unknown."""


@dataclass(frozen=True)
class SearchConfig:
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


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def doc_hash(text: str, meta: dict[str, Any], hash_keys: Sequence[str] = ()) -> str:
    """content_hash of the text, plus the named meta fields when any are given."""
    if not hash_keys:
        return content_hash(text)
    return content_hash(text + "\n" + json.dumps({k: meta.get(k) for k in hash_keys}, sort_keys=True))


def similarity(distance: float, space: str) -> float:
    # Chroma's l2 is squared euclidean; for the unit vectors bge emits,
    # ||a-b||^2 = 2 - 2cos, so cos = 1 - d/2. cosine and ip report 1 - cos.
    if space == "l2":
        return 1.0 - distance / 2.0
    return 1.0 - distance


def _base(cfg: SearchConfig) -> str:
    return f"{cfg.chroma_url.rstrip('/')}/api/v1/collections"


async def _json(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> tuple[int, Any]:
    try:
        resp = await client.request(method, url, **kwargs)
        return resp.status_code, resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchUnavailableError(f"chroma unavailable: {type(exc).__name__}") from exc


async def embed(
    client: httpx.AsyncClient, cfg: SearchConfig, text: str, *, doc_prefix: str,
) -> tuple[list[float], str | None]:
    req = EmbeddingGenerateV1(doc_id=f"{doc_prefix}-{uuid4()}", text=text)
    try:
        resp = await client.post(cfg.embed_url, json=req.model_dump(mode="json"))
        resp.raise_for_status()
        result = EmbeddingResultV1.model_validate(resp.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchUnavailableError(f"embedder unavailable: {type(exc).__name__}") from exc
    if not result.embedding:
        raise SearchUnavailableError("embedder returned no vector")
    return result.embedding, result.embedding_model


async def _collection(client: httpx.AsyncClient, cfg: SearchConfig) -> _Collection | None:
    status, body = await _json(client, "GET", f"{_base(cfg)}/{cfg.collection}")
    if status == 200 and isinstance(body, dict) and body.get("id"):
        space = str((body.get("metadata") or {}).get("hnsw:space") or "l2")
        return _Collection(id=str(body["id"]), space=space)
    # Chroma 0.4.24 answers a missing collection with HTTP 500 + ValueError text.
    error = str(body.get("error") or "") if isinstance(body, dict) else ""
    if status == 500 and f"Collection {cfg.collection} does not exist" in error:
        return None
    raise SearchUnavailableError(f"chroma collection lookup failed: HTTP {status}")


async def _count(client: httpx.AsyncClient, cfg: SearchConfig, coll: _Collection) -> int:
    status, body = await _json(client, "GET", f"{_base(cfg)}/{coll.id}/count")
    if status != 200 or isinstance(body, bool) or not isinstance(body, int):
        raise SearchUnavailableError(f"chroma count failed: HTTP {status}")
    return body


async def nearest(
    client: httpx.AsyncClient, cfg: SearchConfig, vector: list[float], n: int,
    *, where: dict[str, Any] | None = None,
) -> list[tuple[str, float]]:
    """(doc_id, similarity) for the n nearest indexed records, best first.

    ``where`` is a Chroma metadata filter applied before the n nearest are taken.
    """
    coll = await _collection(client, cfg)
    if coll is None:
        raise SearchUnavailableError(f"search index {cfg.collection} not built yet")
    query: dict[str, Any] = {"query_embeddings": [vector], "n_results": n, "include": ["distances"]}
    if where:
        query["where"] = where
    status, body = await _json(client, "POST", f"{_base(cfg)}/{coll.id}/query", json=query)
    if status != 200:
        raise SearchUnavailableError(f"chroma query failed: HTTP {status}")
    try:
        ids, distances = body["ids"][0], body["distances"][0]
        if len(ids) != len(distances) or any(not isinstance(i, str) for i in ids):
            raise ValueError("ids/distances mismatch")
        scored = [(i, similarity(float(d), coll.space)) for i, d in zip(ids, distances)]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SearchUnavailableError("malformed chroma query reply") from exc
    # Chroma clamps n_results to the index size, so an unfiltered query answering
    # nothing means no vectors: unbuilt, not "no match". A filtered one may
    # legitimately match nothing, so ask whether the collection holds anything.
    if not scored and (not where or await _count(client, cfg, coll) == 0):
        raise SearchUnavailableError(f"search index {cfg.collection} is empty")
    return sorted(scored, key=lambda s: -s[1])


async def stored_hashes(client: httpx.AsyncClient, cfg: SearchConfig, ids: list[str]) -> dict[str, str]:
    """content_hash per already-indexed id; a missing collection holds nothing."""
    coll = await _collection(client, cfg)
    if coll is None:
        return {}
    status, body = await _json(
        client, "POST", f"{_base(cfg)}/{coll.id}/get", json={"ids": ids, "include": ["metadatas"]},
    )
    got = body.get("ids") if isinstance(body, dict) else None
    metas = body.get("metadatas") if isinstance(body, dict) else None
    if status != 200 or not isinstance(got, list) or not isinstance(metas, list) or len(got) != len(metas):
        raise SearchUnavailableError(f"chroma get failed: HTTP {status}")
    return {str(i): str((m or {}).get("content_hash") or "") for i, m in zip(got, metas)}


async def publish_upsert(
    bus: Any, source: Any, cfg: SearchConfig, *,
    doc_id: str, text: str, vector: list[float], model: str | None, meta: dict[str, Any],
) -> None:
    payload = VectorUpsertV1(
        doc_id=doc_id, collection=cfg.collection, embedding=vector, embedding_kind="semantic",
        embedding_model=model, embedding_dim=len(vector), text=text, meta=meta,
    )
    await bus.publish(UPSERT_CHANNEL, BaseEnvelope(
        kind=UPSERT_KIND, source=source, payload=payload.model_dump(mode="json"),
    ))


async def index_docs(
    docs: Sequence[tuple[str, str, dict[str, Any]]],
    cfg: SearchConfig,
    *,
    client: httpx.AsyncClient,
    bus: Any,
    source: Any,
    doc_prefix: str,
    batch: int | None = None,
    hash_keys: Sequence[str] = (),
) -> IndexPass:
    """Embed and upsert (doc_id, text, meta) docs the index lacks or holds stale text for.

    meta gains content_hash; pending counts stale docs left past this batch.
    ``hash_keys`` names meta fields folded into the hash, so a change to one
    (e.g. a timestamp a search filters on) re-upserts the doc like a text edit.
    """
    if not docs:
        return IndexPass(indexed=0, pending=0)
    hashed = [(did, text, meta, doc_hash(text, meta, hash_keys)) for did, text, meta in docs]
    stored = await stored_hashes(client, cfg, [d[0] for d in hashed])
    stale = [d for d in hashed if stored.get(d[0]) != d[3]]
    batch = cfg.index_batch if batch is None else batch
    for did, text, meta, digest in stale[:batch]:
        vector, model = await embed(client, cfg, text, doc_prefix=doc_prefix)
        await publish_upsert(
            bus, source, cfg, doc_id=did, text=text, vector=vector, model=model,
            meta={**meta, "content_hash": digest},
        )
    done = min(len(stale), batch)
    return IndexPass(indexed=done, pending=len(stale) - done)
