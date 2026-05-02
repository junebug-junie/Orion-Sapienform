from __future__ import annotations

import logging
from orion.core.bus.bus_schemas import BaseEnvelope

from .settings import get_settings

logger = logging.getLogger("orion.cortex.memory_extractor")


async def handle_memory_history_turn(env: BaseEnvelope) -> None:
    settings = get_settings()
    if not settings.orion_auto_extractor_enabled:
        return
    if settings.orion_auto_extractor_stage2_enabled:
        raise NotImplementedError("ORION_AUTO_EXTRACTOR_STAGE2_ENABLED is v1.5-only")
    logger.debug("memory_extractor noop kind=%s", getattr(env, "kind", ""))
