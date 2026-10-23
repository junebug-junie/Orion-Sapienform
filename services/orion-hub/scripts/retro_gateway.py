"""Dedicated Retro relay process: realtime game traffic off Hub's shared event loop.

Live 2026-10-20 (evals/retro_play.mjs A/B, a Hub page open in another tab): via the
Hub process, input acknowledgments took 720 ms p50 and play dropped 4 times; via this
process, 17 ms and no drops. Same router, origin check and credentials as Hub's own
/api/retro routes; tailscale serve sends /api/retro here (docs/runbooks/retro-browser.md).

Co-play session ends are exported from HERE (scripts/retro_routes.py watch_coplay runs in
the process that relayed the handoff). Until 2026-10-23 only Hub's main.py registered the
exporter, so every co-play handoff through this gateway ended with no session row and no
memory card (found by evals/retro_coplay.mjs: 7 handoffs, 0 rows).

The co-play ``play`` act (actor ``hub.retro_coplay``) is published from here too, so this process
opens its own bus client (ORION_BUS_URL, in the background like the pool). Until 2026-10-23 the
gateway had no bus and co-play opened no act, so ``grammar_acts`` held 0 retro acts while every
played session went through this process.
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
        bus_provider=lambda: getattr(app.state, 'retro_act_bus', None),
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


def register_session_start_sink(app, settings, *, bus_factory=None):
    """Register the co-play act opener now and connect its bus in the background (the relay never
    waits for Redis). An act before the bus is ready is not published (``act_emit`` returns False
    with no bus). Returns the bus task, or None when the bus is disabled or has no URL."""
    from scripts.retro_memory_export import make_session_start_sink
    app.state.retro_act_bus = None
    app.state.retro_session_start_sink = make_session_start_sink(
        bus_provider=lambda: getattr(app.state, 'retro_act_bus', None))
    url = str(getattr(settings, 'ORION_BUS_URL', '') or '').strip()
    if not getattr(settings, 'ORION_BUS_ENABLED', True) or not url:
        logger.warning('retro_gateway_act_bus_skipped reason=bus_disabled_or_unset')
        return None

    async def open_bus():
        try:
            nonlocal bus_factory
            if bus_factory is None:
                from orion.core.bus.async_service import OrionBusAsync
                bus_factory = lambda u: OrionBusAsync(u, enabled=True)  # noqa: E731
            bus = bus_factory(url)
            await bus.connect()
            app.state.retro_act_bus = bus
        except Exception as exc:  # noqa: BLE001 -- the relay keeps running without play acts
            logger.warning('retro_gateway_act_bus_failed error=%s', exc)
        logger.info('retro_gateway_act_bus_ready bus=%s', app.state.retro_act_bus is not None)

    import asyncio
    return asyncio.create_task(open_bus())


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from app.settings import settings
    task = register_session_end_sink(app, settings)
    # No exporter, no close: never open an act that only its horizon could end.
    bus_task = (register_session_start_sink(app, settings)
                if getattr(app.state, 'retro_session_end_sink', None) is not None else None)
    try:
        yield
    finally:
        for t in (task, bus_task):
            if t is not None and not t.done():
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
        act_bus = getattr(app.state, 'retro_act_bus', None)
        if act_bus is not None:
            try:
                await asyncio.wait_for(act_bus.close(), 5)
            except Exception:  # noqa: BLE001
                logger.warning('retro_gateway_act_bus_close_failed', exc_info=True)
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
