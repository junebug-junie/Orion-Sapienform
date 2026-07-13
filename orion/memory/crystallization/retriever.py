from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

from orion.memory.crystallization.active_packet import build_active_packet
from orion.memory.crystallization.chroma_query import query_chroma_collection
from orion.memory.crystallization.dynamics import recall_boost
from orion.memory.crystallization.projection_graphiti import GraphitiAdapter
from orion.memory.crystallization.repository import get_crystallization, update_crystallization
from orion.memory.crystallization.schemas import ActiveMemoryPacketV1, MemoryCrystallizationV1

logger = logging.getLogger(__name__)


async def _persist_recall_boost(
    pool: "asyncpg.Pool",
    crystallization_id: str,
    *,
    now: datetime,
) -> None:
    try:
        current = await get_crystallization(pool, crystallization_id)
        if current is None:
            return
        updated = recall_boost(current, now=now)
        await update_crystallization(pool, updated)
    except Exception as exc:
        logger.warning(
            "retriever_recall_boost_failed crystallization_id=%s error=%s", crystallization_id, exc
        )


async def _apply_recall_boost(
    *,
    pool: "asyncpg.Pool | None",
    crystallization_refs: list[str],
    now: datetime,
) -> None:
    """Reinforce every crystallization that actually made it into the render budget.

    Recall is a weaker signal than recurrence (see `dynamics.recall_boost`'s docstring):
    only what appears in `crystallization_refs` -- not every candidate considered --
    counts as "this got recalled". Refetches each row immediately before mutating,
    rather than reusing the in-memory snapshot collected at the start of retrieval --
    `update_crystallization()` does a full-row UPDATE (existing pattern, matches
    `reinforce()`'s call sites), so persisting from a stale snapshot risks clobbering a
    concurrent governance write. Refetching narrows that staleness window from "however
    long this whole retrieval took" (embed/chroma/graphiti round trips included) down to
    "between this refetch and this write" -- it does not eliminate the race (no
    transaction/row-lock), just shrinks it to the cheapest fix available without
    changing `repository.py`'s persistence contract. Persists concurrently since each
    row's update is independent. Degrades gracefully to a no-op when `pool` is absent
    (e.g. offline/test callers), never raises.
    """
    if pool is None or not crystallization_refs:
        return
    await asyncio.gather(
        *(_persist_recall_boost(pool, cid, now=now) for cid in crystallization_refs)
    )


async def _embed_query(
    query: str,
    *,
    embed_host_url: str,
    embed_timeout_ms: int = 8000,
) -> Optional[list[float]]:
    if not embed_host_url.strip():
        return None
    try:
        import httpx
        from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1

        req = EmbeddingGenerateV1(doc_id="retrieval_query", text=query, embedding_profile="default")
        async with httpx.AsyncClient(timeout=embed_timeout_ms / 1000.0) as client:
            resp = await client.post(embed_host_url.rstrip("/"), json=req.model_dump(mode="json"))
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return EmbeddingResultV1.model_validate(data).embedding
    except Exception as exc:
        logger.warning("retriever_embed_query_failed error=%s", exc)
    return None


async def retrieve_active_packet(
    *,
    query: str,
    crystallizations: list[MemoryCrystallizationV1],
    card_refs: list[str] | None = None,
    active_cards: list[dict[str, Any]] | None = None,
    task_type: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    chroma_host: str = "",
    chroma_port: int = 8000,
    chroma_collection: str = "orion_memory_crystallizations",
    embed_host_url: str = "",
    graphiti_adapter: GraphitiAdapter | None = None,
    seed_crystallization_id: str | None = None,
    pool: "asyncpg.Pool | None" = None,
) -> ActiveMemoryPacketV1:
    """Multi-rail retrieval: Postgres crystallizations + cards + Chroma + Graphiti."""
    chroma_hits: list[dict[str, Any]] = []
    graphiti_refs: list[str] = []
    chroma_refs: list[str] = []

    embedding = await _embed_query(query, embed_host_url=embed_host_url)
    if embedding and chroma_host.strip():
        chroma_hits = query_chroma_collection(
            host=chroma_host,
            port=chroma_port,
            collection=chroma_collection,
            query_embedding=embedding,
        )
        chroma_refs = [str(h.get("doc_id")) for h in chroma_hits if h.get("doc_id")]

    extra_crystallization_ids = set()
    for hit in chroma_hits:
        meta = hit.get("metadata") or {}
        cid = meta.get("crystallization_id")
        if cid:
            extra_crystallization_ids.add(str(cid))

    graphiti_rails: list[str] = []
    if graphiti_adapter and graphiti_adapter.enabled and seed_crystallization_id:
        health = graphiti_adapter.health()
        if health.get("backend") == "graphiti_core":
            graphiti_rails.append("graphiti_search")
            sr = graphiti_adapter.search(query, seed_crystallization_id=seed_crystallization_id)
            for cid in sr.get("crystallization_ids") or []:
                cid_str = str(cid)
                if cid_str not in graphiti_refs:
                    graphiti_refs.append(cid_str)
                extra_crystallization_ids.add(cid_str)
            graphiti_rails.append("graphiti_neighborhood")
            nb = graphiti_adapter.neighborhood(seed_crystallization_id, depth=2)
            for node in nb.get("nodes") or []:
                nid = node.get("crystallization_id") or node.get("id")
                if nid:
                    nid_str = str(nid)
                    if nid_str not in graphiti_refs:
                        graphiti_refs.append(nid_str)
                    extra_crystallization_ids.add(nid_str)
        else:
            graphiti_rails.append("graphiti_neighborhood")
            nb = graphiti_adapter.neighborhood(seed_crystallization_id, depth=2)
            for node in nb.get("nodes") or []:
                nid = node.get("crystallization_id") or node.get("id")
                if nid:
                    graphiti_refs.append(str(nid))
                    extra_crystallization_ids.add(str(nid))

    merged = {c.crystallization_id: c for c in crystallizations}
    packet = build_active_packet(
        query=query,
        crystallizations=list(merged.values()),
        card_refs=card_refs,
        active_cards=active_cards,
        task_type=task_type,
        project_id=project_id,
        session_id=session_id,
    )

    packet.chroma_refs = chroma_refs
    packet.graphiti_refs = graphiti_refs
    trace = dict(packet.retrieval_trace)
    rails = list(trace.get("rails") or [])
    rails.extend(["chroma_semantic"])
    rails.extend(graphiti_rails)
    trace["rails"] = rails
    trace["chroma_hits"] = len(chroma_hits)
    trace["graphiti_refs"] = len(graphiti_refs)
    trace["extra_crystallization_ids_from_chroma"] = sorted(extra_crystallization_ids)
    packet.retrieval_trace = trace

    await _apply_recall_boost(
        pool=pool,
        crystallization_refs=packet.crystallization_refs,
        now=datetime.now(timezone.utc),
    )

    return packet
