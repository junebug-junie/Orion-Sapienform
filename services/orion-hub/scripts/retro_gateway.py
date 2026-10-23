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


async def register_session_end_sink(app, settings, *, create_pool=None) -> None:
    """Open the memory pool (RECALL_PG_DSN, as Hub does) and register the co-play exporter.
    No DSN: no sink, logged; play is unaffected."""
    from scripts.retro_memory_export import make_session_end_sink
    app.state.memory_pg_pool = None
    dsn = str(getattr(settings, 'RECALL_PG_DSN', '') or '').strip()
    if not dsn:
        logger.warning('retro_gateway_session_end_sink_skipped reason=RECALL_PG_DSN_unset')
        return
    try:
        if create_pool is None:
            import asyncpg
            create_pool = asyncpg.create_pool
        from scripts.memory_pg_pool import create_memory_pg_pool_with_retry
        app.state.memory_pg_pool = await create_memory_pg_pool_with_retry(dsn=dsn, create_pool=create_pool)
    except Exception as exc:  # noqa: BLE001 -- the relay must start even without the memory DB
        logger.warning('retro_gateway_memory_pool_failed error=%s', exc)
    # Registered even without a pool: export_session then logs "no_memory_pool" instead of the
    # sink being silently absent.
    app.state.retro_session_end_sink = make_session_end_sink(
        pool_provider=lambda: getattr(app.state, 'memory_pg_pool', None),
        bus_provider=lambda: None,   # co-play has no play act to close
    )
    logger.info('retro_gateway_session_end_sink_ready pool=%s', app.state.memory_pg_pool is not None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.settings import settings
    await register_session_end_sink(app, settings)
    try:
        yield
    finally:
        pool = getattr(app.state, 'memory_pg_pool', None)
        if pool is not None:
            await pool.close()


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.include_router(configured_router())


@app.get('/api/retro-gateway/health')
async def health():
    return {'status': 'ok'}
