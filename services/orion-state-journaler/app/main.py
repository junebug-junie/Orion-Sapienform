from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import List

import asyncpg
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from .service import StateJournaler
from .settings import settings


journaler = StateJournaler()


async def _fetch_rollups(window: int, hours: int) -> List[asyncpg.Record]:
    sql = f"""
    SELECT bucket_ts, window_sec, node, avg_valence, avg_arousal, avg_coherence, avg_novelty, pct_missing, pct_stale, avg_distress
    FROM {settings.rollup_table}
    WHERE window_sec=$1 AND bucket_ts >= (NOW() - INTERVAL '{int(hours)} hours')
    ORDER BY bucket_ts DESC
    """
    conn = await asyncpg.connect(dsn=settings.postgres_uri)
    try:
        rows = await conn.fetch(sql, int(window))
        return rows
    finally:
        await conn.close()


app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(journaler.start())


@app.on_event("shutdown")
async def _shutdown() -> None:
    await journaler.stop()


@app.get("/rollups")
async def get_rollups(
    window: int = Query(300, description="Window size in seconds"),
    hours: int = Query(24, description="Lookback horizon in hours"),
):
    rows = await _fetch_rollups(window, hours)
    out = [
        {
            "bucket_ts": r["bucket_ts"].isoformat() if r.get("bucket_ts") else None,
            "window_sec": r.get("window_sec"),
            "node": r.get("node"),
            "avg_valence": r.get("avg_valence"),
            "avg_arousal": r.get("avg_arousal"),
            "avg_coherence": r.get("avg_coherence"),
            "avg_novelty": r.get("avg_novelty"),
            "pct_missing": r.get("pct_missing"),
            "pct_stale": r.get("pct_stale"),
            "avg_distress": r.get("avg_distress"),
        }
        for r in rows
    ]
    return JSONResponse({"window": window, "hours": hours, "rows": out})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
