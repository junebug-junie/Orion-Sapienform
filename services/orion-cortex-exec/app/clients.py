from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from pydantic import ValidationError
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ServiceRef
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
        Sends a typed ChatRequestPayload, returns a validated result dict.
        (Ideally, we would return a ChatResponsePayload here, but for now 
        we at least ensure the request is strictly typed).
        """
        
        # 1. STRICT: We only accept a Pydantic model as input
        payload_json = req.model_dump(mode="json")

        env = BaseEnvelope(
            kind="llm.chat.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=payload_json,
        )

        logger.debug(f"Client sending RPC to {self.channel} (timeout={self.timeout}s)")

        # 2. LOOSE: The bus is the only place we touch raw serialization
        msg = await self.bus.rpc_request(
            self.channel, 
            env, 
            reply_channel=reply_to, 
            timeout_sec=self.timeout
        )

        # 3. RE-STRICT: Validate response envelope
        decoded = self.bus.codec.decode(msg.get("data"))
        
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        
        # In a perfect world, we would do: ChatResponsePayload.model_validate(decoded.envelope.payload)
        # For now, we return the dict, but we know it came from a valid envelope.
        if not isinstance(decoded.envelope.payload, dict):
             raise ValueError(f"Expected dict payload, got {type(decoded.envelope.payload)}")
             
        return decoded.envelope.payload
