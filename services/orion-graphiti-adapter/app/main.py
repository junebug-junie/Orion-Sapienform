import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.backends import graphiti_core as core_backend
from app.backends import orion_postgres as pg_backend
from app.falkordb import sync_to_falkordb
from app.settings import settings
from app.store import apply_graphiti_schema

logger = logging.getLogger(settings.SERVICE_NAME)
pg_pool: Optional[asyncpg.Pool] = None


class CrystallizationLinkIngestV1(BaseModel):
    target_crystallization_id: str
    relation: str
    confidence: float = 0.5


class EpisodeIngestV1(BaseModel):
    crystallization_id: str
    kind: str
    subject: str
    summary: str
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    links: list[CrystallizationLinkIngestV1] = Field(default_factory=list)


class SearchRequestV1(BaseModel):
    query: str
    seed_crystallization_id: str | None = None
    limit: int = 10


class RebuildItemV1(EpisodeIngestV1):
    pass


class RebuildRequestV1(BaseModel):
    items: list[RebuildItemV1]


def _backend():
    return core_backend if settings.GRAPHITI_BACKEND == "graphiti_core" else pg_backend


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pg_pool
    dsn = (settings.POSTGRES_URI or "").strip()
    if dsn:
        pg_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)
        if settings.GRAPHITI_AUTO_APPLY_SCHEMA:
            apply_graphiti_schema(dsn)
    if (
        settings.GRAPHITI_BACKEND == "graphiti_core"
        and settings.FALKORDB_ENABLED
        and settings.GRAPHITI_AUTO_BUILD_INDICES
    ):
        await core_backend.ensure_graphiti_indices(settings.FALKORDB_URI, settings.FALKORDB_GRAPH)
    app.state.pg_pool = pg_pool
    yield
    if pg_pool:
        await pg_pool.close()


app = FastAPI(title="Orion Graphiti Adapter", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "service": settings.SERVICE_NAME,
        "postgres": pg_pool is not None,
        "falkordb_enabled": settings.FALKORDB_ENABLED,
        "backend": settings.GRAPHITI_BACKEND,
    }


@app.post("/v1/episodes")
async def ingest_episode(body: EpisodeIngestV1) -> dict:
    if body.metadata.get("sensitivity") == "intimate":
        return {"skipped": True, "reason": "intimate_sensitivity", "canonical_mutated": False}

    backend = _backend()
    episode_id = f"gep_{body.crystallization_id}"
    link_payload = [l.model_dump() for l in body.links]
    falkor_result = None

    if settings.GRAPHITI_BACKEND == "graphiti_core":
        if pg_pool is None:
            raise HTTPException(status_code=503, detail="store_unavailable")
        result = await backend.ingest_episode(
            pg_pool,
            episode_id=episode_id,
            crystallization_id=body.crystallization_id,
            kind=body.kind,
            subject=body.subject,
            summary=body.summary,
            status=body.status,
            metadata=body.metadata,
            links=link_payload,
            falkordb_uri=settings.FALKORDB_URI,
            graph_name=settings.FALKORDB_GRAPH,
            embed_url=settings.CRYSTALLIZER_EMBED_HOST_URL,
        )
        if result.get("skipped"):
            return {"skipped": True, "reason": result.get("reason", "intimate_sensitivity"), "canonical_mutated": False}
    else:
        if pg_pool is None:
            raise HTTPException(status_code=503, detail="store_unavailable")
        result = await backend.ingest_episode(
            pg_pool,
            episode_id=episode_id,
            crystallization_id=body.crystallization_id,
            kind=body.kind,
            subject=body.subject,
            summary=body.summary,
            status=body.status,
            metadata=body.metadata,
            links=link_payload,
        )
        if settings.FALKORDB_ENABLED:
            falkor_result = sync_to_falkordb(
                uri=settings.FALKORDB_URI,
                graph_name=settings.FALKORDB_GRAPH,
                crystallization_id=body.crystallization_id,
                kind=body.kind,
                subject=body.subject,
                summary=body.summary,
                links=link_payload,
            )

    edge_ids = result["edge_ids"]
    return {
        "episode_id": episode_id,
        "entity_id": f"gent_{body.crystallization_id}",
        "edge_id": edge_ids[0] if edge_ids else f"ged_{body.crystallization_id}",
        "edge_ids": edge_ids,
        "falkordb": falkor_result,
        "canonical_mutated": False,
    }


@app.post("/v1/rebuild")
async def rebuild(body: RebuildRequestV1) -> dict:
    results: list[dict] = []
    skip_count = 0
    for item in body.items:
        result = await ingest_episode(item)
        if result.get("skipped"):
            skip_count += 1
        else:
            results.append(result)
    return {
        "ingested": len(results),
        "skipped_intimate": skip_count,
        "canonical_mutated": False,
    }


@app.get("/v1/neighborhood/{crystallization_id}")
async def get_neighborhood(crystallization_id: str, depth: int = 1) -> dict:
    if pg_pool is None:
        raise HTTPException(status_code=503, detail="store_unavailable")
    return await _backend().get_neighborhood(pg_pool, crystallization_id, depth=depth)


@app.post("/v1/search")
async def search_episodes(body: SearchRequestV1) -> dict:
    if settings.GRAPHITI_BACKEND != "graphiti_core":
        raise HTTPException(status_code=501, detail="search_requires_graphiti_core_backend")
    return await core_backend.search(
        body.query,
        seed_crystallization_id=body.seed_crystallization_id or "",
        limit=body.limit,
        embed_url=settings.CRYSTALLIZER_EMBED_HOST_URL,
        falkordb_uri=settings.FALKORDB_URI,
        graph_name=settings.FALKORDB_GRAPH,
    )
