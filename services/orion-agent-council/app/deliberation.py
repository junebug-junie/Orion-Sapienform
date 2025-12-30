# services/orion-agent-council/app/deliberation.py
from __future__ import annotations

import logging
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync

from .models import DeliberationRequest
from .pipeline import build_default_pipeline

logger = logging.getLogger("agent-council.deliberation")


class DeliberationRouter:
    """
    Async router/factory.
    """

    def __init__(self, bus: OrionBusAsync) -> None:
        self.bus = bus

    async def handle(self, raw_payload: Dict[str, Any]) -> None:
        try:
            req = DeliberationRequest(**raw_payload)
        except Exception as e:
            logger.error("DeliberationRequest parse error: %s payload=%r", e, raw_payload)
            return

        pipeline = build_default_pipeline(bus=self.bus, req=req)

        logger.info(
            "[%s] Routing council_deliberation (source=%s universe=%s)",
            req.trace_id or "no-trace",
            req.source or "unknown",
            req.universe or "core",
        )

        await pipeline.run(req) # [FIX] Await async pipeline
