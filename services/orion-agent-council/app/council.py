# services/orion-agent-council/app/council.py
from __future__ import annotations

import logging
from typing import Dict, Any

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.async_service import OrionBusAsync

from .deliberation import DeliberationRouter
from .settings import settings

logger = logging.getLogger("agent-council.council")

class CouncilService:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.router = DeliberationRouter(bus)

    async def handle_envelope(self, env: BaseEnvelope) -> None:
        payload = env.payload
        if not isinstance(payload, dict):
            logger.warning("Ignoring non-dict payload from %s", env.source)
            return

        event = payload.get("event")
        if event and event != "council_deliberation":
            logger.debug("Ignoring non-deliberation event: %s", event)
            return

        if not payload.get("trace_id"):
            payload["trace_id"] = str(env.correlation_id)

        # [FIX] Natively await async router logic
        try:
            await self.router.handle(payload)
        except Exception as e:
            logger.exception("Error processing council request: %s", e)
