import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.falkordb import sync_to_falkordb
from app.settings import settings
from app.store import apply_graphiti_schema, neighborhood, upsert_episode

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pg_pool
    dsn = (settings.POSTGRES_URI or "").strip()
    if dsn:
        pg_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=4)
        if settings.GRAPHITI_AUTO_APPLY_SCHEMA:
            apply_graphiti_schema(dsn)
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
    }


@app.post("/v1/episodes")
async def ingest_episode(body: EpisodeIngestV1) -> dict:
    if pg_pool is None:
        raise HTTPException(status_code=503, detail="store_unavailable")
    episode_id = f"gep_{body.crystallization_id}"
    edge_ids = await upsert_episode(
        pg_pool,
        episode_id=episode_id,
        crystallization_id=body.crystallization_id,
        kind=body.kind,
        subject=body.subject,
        summary=body.summary,
        status=body.status,
        metadata=body.metadata,
        links=[l.model_dump() for l in body.links],
    )
    falkor_result = None
    if settings.FALKORDB_ENABLED:
        falkor_result = sync_to_falkordb(
            uri=settings.FALKORDB_URI,
            graph_name=settings.FALKORDB_GRAPH,
            crystallization_id=body.crystallization_id,
            kind=body.kind,
            subject=body.subject,
            summary=body.summary,
        )
    return {
        "episode_id": episode_id,
        "entity_id": f"gent_{body.crystallization_id}",
        "edge_id": edge_ids[0] if edge_ids else f"ged_{body.crystallization_id}",
        "edge_ids": edge_ids,
        "falkordb": falkor_result,
        "canonical_mutated": False,
    }


@app.get("/v1/neighborhood/{crystallization_id}")
async def get_neighborhood(crystallization_id: str) -> dict:
    if pg_pool is None:
        raise HTTPException(status_code=503, detail="store_unavailable")
    return await neighborhood(pg_pool, crystallization_id)
