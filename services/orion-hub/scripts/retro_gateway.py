"""Dedicated Retro relay process: realtime game traffic off Hub's shared event loop.

Live 2026-10-20 (evals/retro_play.mjs A/B, a Hub page open in another tab): via the
Hub process, input acknowledgments took 720 ms p50 and play dropped 4 times; via this
process, 17 ms and no drops. Same router, origin check and credentials as Hub's own
/api/retro routes; tailscale serve sends /api/retro here (docs/runbooks/retro-browser.md).

Co-play session ends are exported from HERE (scripts/retro_routes.py watch_coplay runs in
the process that relayed the handoff). Until 2026-10-23 only Hub's main.py registered the
exporter, so every co-play handoff through this gateway ended with no session row and no
memory card (found by evals/retro_coplay.mjs: 7 handoffs, 0 rows).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from scripts.retro_routes import configured_router

logger = logging.getLogger('orion-hub.retro_gateway')


def register_session_end_sink(app, settings, *, create_pool=None):
    """Register the co-play exporter now and open its memory pool (RECALL_PG_DSN, as Hub does) in
    the background: the realtime relay never waits for the database (a sql-db restart would
    otherwise refuse the player's socket for the pool's ~58 s retry). An export before the pool
    is ready logs ``no_memory_pool``. Returns the pool task, or None without a DSN."""
    from scripts.retro_memory_export import make_session_end_sink
    app.state.memory_pg_pool = None
    dsn = str(getattr(settings, 'RECALL_PG_DSN', '') or '').strip()
    if not dsn:
        logger.warning('retro_gateway_session_end_sink_skipped reason=RECALL_PG_DSN_unset')
        return None
    app.state.retro_session_end_sink = make_session_end_sink(
        pool_provider=lambda: getattr(app.state, 'memory_pg_pool', None),
        bus_provider=lambda: None,   # co-play has no play act to close
    )

    async def open_pool():
        try:
            nonlocal create_pool
            if create_pool is None:
                import asyncpg
                create_pool = asyncpg.create_pool
            from scripts.memory_pg_pool import create_memory_pg_pool_with_retry
            app.state.memory_pg_pool = await create_memory_pg_pool_with_retry(dsn=dsn, create_pool=create_pool)
        except Exception as exc:  # noqa: BLE001 -- the relay keeps running without the memory DB
            logger.warning('retro_gateway_memory_pool_failed error=%s', exc)
        logger.info('retro_gateway_session_end_sink_ready pool=%s', app.state.memory_pg_pool is not None)

    import asyncio
    return asyncio.create_task(open_pool())


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from app.settings import settings
    task = register_session_end_sink(app, settings)
    try:
        yield
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        pool = getattr(app.state, 'memory_pg_pool', None)
        if pool is not None:
            try:
                await asyncio.wait_for(pool.close(), 5)
            except Exception:  # noqa: BLE001 -- a checked-out connection must not hold shutdown
                pool.terminate()


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.include_router(configured_router())


@app.get('/api/retro-gateway/health')
async def health():
    return {'status': 'ok'}
