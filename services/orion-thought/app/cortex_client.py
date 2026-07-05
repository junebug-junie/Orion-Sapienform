from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.schemas import PlanExecutionRequest

from .settings import settings

logger = logging.getLogger("orion-thought.cortex")


class CortexExecClient:
    """Typed RPC client for cortex-exec PlanExecutionRequest."""

    def __init__(
        self,
        bus: OrionBusAsync,
        *,
        request_channel: str | None = None,
        result_prefix: str | None = None,
    ) -> None:
        self.bus = bus
        self.request_channel = request_channel or settings.channel_cortex_exec_request
        self.result_prefix = result_prefix or settings.channel_cortex_exec_result_prefix

    async def execute_plan(
        self,
        *,
        source: ServiceRef,
        req: PlanExecutionRequest,
        correlation_id: str,
        timeout_sec: float,
    ) -> dict[str, Any]:
        reply_channel = f"{self.result_prefix}:{uuid4()}"
        env = BaseEnvelope(
            kind=req.kind,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=req.model_dump(mode="json"),
        )
        logger.info(
            "RPC emit -> %s kind=%s corr=%s reply=%s verb=%s",
            self.request_channel,
            env.kind,
            correlation_id,
            reply_channel,
            req.plan.verb_name,
        )
        msg = await self.bus.rpc_request(
            self.request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Exec RPC failed: {decoded.error}")

        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            nested = payload.get("result")
            if isinstance(nested, dict):
                return nested
            return payload
        raise RuntimeError("Exec RPC returned non-dict payload")
