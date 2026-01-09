# ==================================================
# app/context.py — Shared Application Context
# ==================================================
import asyncio
import logging
from typing import Any, Coroutine

from app.settings import settings
from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("dream-app.context")

# This global instance will be shared
bus: OrionBusAsync | None = None


def _run_async(coro: asyncio.Future | Coroutine[Any, Any, Any]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    loop.create_task(coro)


def initialize_bus() -> None:
    """Initializes the global bus instance."""
    global bus
    if settings.ORION_BUS_ENABLED:
        logger.info(f"Initializing OrionBusAsync connection to {settings.ORION_BUS_URL}")
        bus = OrionBusAsync(url=settings.ORION_BUS_URL)
        _run_async(bus.connect())
    else:
        logger.warning("OrionBusAsync is disabled. No messages will be published.")
