from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import asyncpg
import httpx

try:
    from app.settings import settings
except ImportError:  # pragma: no cover
    from settings import settings  # type: ignore

logger = logging.getLogger("orion-recall.cards.embedding")

_EMBED_SUBKEY = "recall_embedding"


def resolve_cards_embedding_url() -> Optional[str]:
    raw = (
        (getattr(settings, "RECALL_CARDS_EMBEDDING_URL", None) or "")
        or (getattr(settings, "RECALL_VECTOR_EMBEDDING_URL", None) or "")
    ).strip()
    if raw:
        return raw.rstrip("/")
    host = (os.getenv("VECTOR_HOST") or "orion-athena-vector-host").strip()
    port = (os.getenv("VECTOR_HOST_PORT") or "8320").strip()
    if not host:
        return None
    return f"http://{host}:{port}/embedding"


def card_recall_text(row: asyncpg.Record | Dict[str, Any]) -> str:
    """Canonical embeddable text for a memory card (title + summary + anchors/tags)."""
    if isinstance(row, dict):
        title = str(row.get("title") or "")
        summary = str(row.get("summary") or "")
        tags = row.get("tags")
        anchors = row.get("anchors")
        ac = row.get("anchor_class")
    else:
        title = str(row["title"] or "")
        summary = str(row["summary"] or "")
        tags = row["tags"]
        anchors = row["anchors"]
        ac = row["anchor_class"]
    parts = [title, summary]
    if ac:
        parts.append(f"anchor:{ac}")
    for t in tags or []:
        if t:
            parts.append(f"tag:{t}")
    for a in anchors or []:
        if a:
            parts.append(f"anchor:{a}")
    return " · ".join(p.strip() for p in parts if p.strip())


def text_fingerprint(text: str) -> str:
    norm = " ".join((text or "").lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def read_cached_embedding(subschema_raw: Any, *, text_fp: str) -> Optional[List[float]]:
    sub: Dict[str, Any]
    if isinstance(subschema_raw, str):
        try:
            sub = json.loads(subschema_raw)
        except Exception:
            return None
    elif isinstance(subschema_raw, dict):
        sub = subschema_raw
    else:
        return None
    block = sub.get(_EMBED_SUBKEY)
    if not isinstance(block, dict):
        return None
    if str(block.get("text_fp") or "") != text_fp:
        return None
    vec = block.get("vector")
    if not isinstance(vec, list) or not vec:
        return None
    return [float(v) for v in vec]


async def _embed_one(
    client: httpx.AsyncClient,
    *,
    url: str,
    text: str,
) -> Tuple[List[float], Optional[str], Optional[int]]:
    payload = {
        "doc_id": f"cards-recall-{uuid4()}",
        "text": text,
        "embedding_profile": "default",
        "include_latent": False,
    }
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    vec = data.get("embedding")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("embedding_response_missing_vector")
    model = data.get("embedding_model")
    dim = data.get("embedding_dim")
    return [float(v) for v in vec], (str(model) if model else None), (int(dim) if dim else len(vec))


async def embed_texts(
    texts: List[str],
    *,
    url: Optional[str] = None,
    timeout_sec: float = 5.0,
    max_concurrency: int = 4,
) -> Dict[str, List[float]]:
    """Embed unique non-empty texts; returns mapping text -> vector."""
    endpoint = (url or resolve_cards_embedding_url() or "").strip()
    unique = [t for t in dict.fromkeys(t.strip() for t in texts if (t or "").strip())]
    if not endpoint or not unique:
        return {}
    out: Dict[str, List[float]] = {}
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    timeout = httpx.Timeout(timeout_sec)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def _one(text: str) -> None:
            async with sem:
                vec, _, _ = await _embed_one(client, url=endpoint, text=text)
                out[text] = vec

        await asyncio.gather(*(_one(t) for t in unique))
    return out


async def persist_card_embeddings(
    pool: asyncpg.Pool,
    updates: List[Tuple[Any, Dict[str, Any], List[float], str, Optional[str], Optional[int], str]],
) -> None:
    """Batch-write recall_embedding blocks into subschema."""
    if not updates:
        return
    async with pool.acquire() as conn:
        async with conn.transaction():
            for card_id, subschema, vector, text_fp, model, dim, _text in updates:
                sub = dict(subschema or {})
                sub[_EMBED_SUBKEY] = {
                    "text_fp": text_fp,
                    "vector": vector,
                    "model": model,
                    "dim": dim or len(vector),
                }
                await conn.execute(
                    """
                    UPDATE memory_cards
                    SET subschema = $2::jsonb, updated_at = now()
                    WHERE card_id = $1
                    """,
                    card_id,
                    json.dumps(sub),
                )


async def score_cards_by_embedding(
    pool: asyncpg.Pool,
    rows: List[asyncpg.Record],
    *,
    query_text: str,
    min_similarity: float,
) -> List[Tuple[float, asyncpg.Record]]:
    """Return (cosine_similarity, row) pairs above min_similarity, sorted desc."""
    q = (query_text or "").strip()
    if not q or not rows:
        return []

    endpoint = resolve_cards_embedding_url()
    if not endpoint:
        logger.warning("cards embedding url unset; vector scoring skipped")
        return []

    timeout = float(getattr(settings, "RECALL_CARDS_EMBED_TIMEOUT_SEC", 5.0) or 5.0)
    concurrency = int(getattr(settings, "RECALL_CARDS_EMBED_CONCURRENCY", 4) or 4)
    card_texts: Dict[str, str] = {}
    cached: Dict[str, List[float]] = {}
    pending_texts: List[str] = []
    pending_meta: List[Tuple[str, str, asyncpg.Record, Dict[str, Any], str]] = []

    for row in rows:
        text = card_recall_text(row)
        fp = text_fingerprint(text)
        cid = str(row["card_id"])
        card_texts[cid] = text
        sub_raw = row["subschema"] if "subschema" in row.keys() else {}
        sub: Dict[str, Any]
        if isinstance(sub_raw, str):
            try:
                sub = json.loads(sub_raw)
            except Exception:
                sub = {}
        elif isinstance(sub_raw, dict):
            sub = dict(sub_raw)
        else:
            sub = {}
        hit = read_cached_embedding(sub, text_fp=fp)
        if hit:
            cached[cid] = hit
        else:
            pending_texts.append(text)
            pending_meta.append((cid, text, row, sub, fp))

    try:
        q_vecs = await embed_texts([q], url=endpoint, timeout_sec=timeout, max_concurrency=concurrency)
    except Exception as exc:
        logger.warning("cards query embed failed: %s", exc)
        return []
    q_vec = q_vecs.get(q)
    if not q_vec:
        return []

    new_vectors: Dict[str, List[float]] = {}
    if pending_texts:
        try:
            new_vectors = await embed_texts(
                pending_texts, url=endpoint, timeout_sec=timeout, max_concurrency=concurrency
            )
        except Exception as exc:
            logger.warning("cards batch embed failed: %s", exc)

    persist_batch: List[Tuple[Any, Dict[str, Any], List[float], str, Optional[str], Optional[int], str]] = []
    scored: List[Tuple[float, asyncpg.Record]] = []

    for row in rows:
        cid = str(row["card_id"])
        text = card_texts[cid]
        vec = cached.get(cid) or new_vectors.get(text)
        if not vec:
            continue
        sim = cosine_similarity(q_vec, vec)
        if sim < min_similarity:
            continue
        scored.append((sim, row))
        if cid not in cached and text in new_vectors:
            sub_raw = row["subschema"] if "subschema" in row.keys() else {}
            sub = dict(sub_raw) if isinstance(sub_raw, dict) else {}
            persist_batch.append((row["card_id"], sub, vec, text_fingerprint(text), None, len(vec), text))

    if persist_batch:
        try:
            await persist_card_embeddings(pool, persist_batch)
        except Exception as exc:
            logger.debug("cards embedding persist skipped: %s", exc)

    scored.sort(key=lambda x: (-x[0], str(x[1]["updated_at"] or "")))
    return scored
