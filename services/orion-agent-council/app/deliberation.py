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

    async def handle(
        self,
        raw_payload: Dict[str, Any],
        *,
        reply_to: str | None,
        correlation_id: str | None,
    ) -> None:
        try:
            req = DeliberationRequest(**raw_payload)
        except Exception as e:
            logger.error(f"DeliberationRequest parse error: {e} payload={raw_payload!r}")
            return

        pipeline = build_default_pipeline(
            bus=self.bus,
            req=req,
            reply_to=reply_to,
            correlation_id=correlation_id,
        )

        logger.info(
            f"[{req.trace_id or 'no-trace'}] Routing council_deliberation "
            f"(source={req.source or 'unknown'} universe={req.universe or 'core'})"
        )

        await pipeline.run(req) # [FIX] Await async pipeline
