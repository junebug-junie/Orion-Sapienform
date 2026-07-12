from __future__ import annotations

import logging

import asyncpg

from orion.memory.crystallization.chroma_query import query_chroma_collection
from orion.memory.crystallization.repository import get_crystallization
from orion.memory.crystallization.retriever import _embed_query
from orion.memory.crystallization.schemas import MemoryCrystallizationV1

logger = logging.getLogger(__name__)


async def fetch_similar_candidates(
    candidate: MemoryCrystallizationV1,
    *,
    pool: asyncpg.Pool,
    embed_host_url: str = "",
    chroma_host: str = "",
    chroma_port: int = 8000,
    chroma_collection: str = "orion_memory_crystallizations",
    limit: int = 5,
    embed_timeout_ms: int = 8000,
) -> list[MemoryCrystallizationV1]:
    """Vector-similarity candidate retrieval across ALL active crystallizations, not scope-gated.

    This exists specifically to fix a confirmed bug: window-scoped Jaccard dedup can never
    match across two different conversation windows. Degrades to [] on any missing config
    or failure at any step — never raises.
    """
    if not embed_host_url.strip() or not chroma_host.strip() or pool is None:
        return []

    query_text = f"{candidate.subject}\n{candidate.summary}".strip()
    if not query_text:
        return []

    try:
        embedding = await _embed_query(
            query_text,
            embed_host_url=embed_host_url,
            embed_timeout_ms=embed_timeout_ms,
        )
    except Exception as exc:
        logger.warning("candidate_retrieval_embed_failed error=%s", exc)
        return []

    if not embedding:
        return []

    try:
        hits = query_chroma_collection(
            host=chroma_host,
            port=chroma_port,
            collection=chroma_collection,
            query_embedding=embedding,
            n_results=max(1, int(limit)),
        )
    except Exception as exc:
        logger.warning("candidate_retrieval_chroma_query_failed error=%s", exc)
        return []

    out: list[MemoryCrystallizationV1] = []
    seen: set[str] = set()

    for hit in hits:
        cid = (hit.get("metadata") or {}).get("crystallization_id") or hit.get("doc_id")
        if not cid:
            continue
        cid = str(cid)
        if cid == candidate.crystallization_id or cid in seen:
            continue
        seen.add(cid)

        try:
            row = await get_crystallization(pool, cid)
        except Exception as exc:
            logger.warning("candidate_retrieval_get_crystallization_failed crystallization_id=%s error=%s", cid, exc)
            continue

        if row is None or row.status != "active":
            continue

        out.append(row)
        if len(out) >= limit:
            break

    return out
