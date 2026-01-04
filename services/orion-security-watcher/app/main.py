import asyncio
import logging

from fastapi import FastAPI

from .bus_worker import bus_worker
from .context import ctx
from .routers import debug, state, test, ui

logger = logging.getLogger("orion-security-watcher")

app = FastAPI(
    title="Orion Security Watcher",
    version=ctx.settings.SERVICE_VERSION,
)

# Routers
app.include_router(ui.router)
app.include_router(state.router)
app.include_router(test.router)
app.include_router(debug.router)


@app.get("/health")
async def health():
    s = ctx.state_store.load()
    return {
        "ok": True,
        "service": ctx.settings.SERVICE_NAME,
        "version": ctx.settings.SERVICE_VERSION,
        "enabled": ctx.settings.SECURITY_ENABLED,
        "armed": s.armed,
        "mode": s.mode,
        "bus_enabled": ctx.bus.enabled,
        "vision_channel": ctx.settings.VISION_EVENTS_SUBSCRIBE_RAW,
    }


@app.on_event("startup")
async def on_startup():
    if ctx.settings.SECURITY_ENABLED and ctx.bus.enabled:
        asyncio.create_task(bus_worker(ctx))
