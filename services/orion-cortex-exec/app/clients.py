from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from pydantic import ValidationError
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ChatResponsePayload, ServiceRef
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.clients")

class LLMGatewayClient:
    """
    Strict, typed client for the LLM Gateway.
    Prevents 'dict soup' by enforcing Pydantic models on both ends.
    """
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.channel = settings.channel_llm_intake
        self.timeout = float(settings.step_timeout_ms) / 1000.0

    async def chat(
        self,
        source: ServiceRef,
        req: ChatRequestPayload,
        correlation_id: str,
        reply_to: str
    ) -> Dict[str, Any]:
        """
        Sends a typed request, returns typed response.
        """

        env = BaseEnvelope(
            kind="llm.chat.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )

        msg = await self.bus.rpc_request(
            self.channel,
            env,
            reply_channel=reply_to,
            timeout_sec=self.timeout
        )

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")

        return ChatResponsePayload.model_validate(decoded.envelope.payload)
