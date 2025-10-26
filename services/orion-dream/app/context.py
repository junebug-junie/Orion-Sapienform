# ==================================================
# app/context.py — Shared Application Context
# ==================================================
import logging
from app.settings import settings
from orion.core.bus.service import OrionBus

logger = logging.getLogger("dream-app.context")

# This global instance will be shared
bus: OrionBus | None = None

def initialize_bus():
    """Initializes the global bus instance."""
    global bus
    if settings.ORION_BUS_ENABLED:
        logger.info(f"Initializing OrionBus connection to {settings.ORION_BUS_URL}")
        bus = OrionBus(url=settings.ORION_BUS_URL)
    else:
        logger.warning("OrionBus is disabled. No messages will be published.")
