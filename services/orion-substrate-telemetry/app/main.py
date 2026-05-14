from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from . import db
from .pg_pool import close_pool, init_pool, pool
from .service import SubstrateTelemetryService
from .settings import settings

svc = SubstrateTelemetryService()


def _optional_auth(x_telemetry_token: str | None = Header(default=None, alias="X-Telemetry-Token")) -> None:
    expected = settings.read_api_token
    if expected:
        if (x_telemetry_token or "").strip() != expected.strip():
            raise HTTPException(status_code=401, detail="unauthorized")


def _row_to_json(r: asyncpg.Record) -> dict[str, Any]:
    return {
        "id": str(r["id"]),
        "correlation_id": str(r["correlation_id"]),
        "envelope_kind": r["envelope_kind"],
        "generated_at": r["generated_at"],
        "cold_anchors": r["cold_anchors"],
        "tier_outcomes": r["tier_outcomes"],
        "degraded_producers": r["degraded_producers"],
        "source_service": r["source_service"],
        "source_node": r["source_node"],
        "received_at_utc": r["received_at_utc"].isoformat() if r["received_at_utc"] else None,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    async with pool().acquire() as conn:
        await db.ensure_schema(conn)
    await svc.start_background()
    try:
        yield
    finally:
        await svc.stop()
        await close_pool()


app = FastAPI(title="orion-substrate-telemetry", lifespan=lifespan)


@app.get("/v1/substrate/tier-outcomes/latest")
async def latest(
    correlation_id: str = Query(..., description="UUID string matching bus envelope"),
    _: None = Depends(_optional_auth),
):
    try:
        cid = UUID(correlation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid_correlation_id")
    async with pool().acquire() as conn:
        await db.ensure_schema(conn)
        row = await db.fetch_latest(conn, correlation_id=cid)
    if row is None:
        raise HTTPException(status_code=404, detail="not_found")
    return JSONResponse(_row_to_json(row))


@app.get("/v1/substrate/tier-outcomes/history")
async def history(
    correlation_id: str = Query(...),
    limit: int = Query(20, ge=1, le=100),
    _: None = Depends(_optional_auth),
):
    try:
        cid = UUID(correlation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid_correlation_id")
    async with pool().acquire() as conn:
        await db.ensure_schema(conn)
        rows = await db.fetch_history(conn, correlation_id=cid, limit=limit)
    return JSONResponse({"correlation_id": str(cid), "items": [_row_to_json(r) for r in rows]})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
