import logging
import uuid
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
from scripts.settings import settings

logger = logging.getLogger("hub.bus.cortex")

class CortexGatewayClient:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self._source = ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
        )

    async def chat(self, request: CortexChatRequest) -> CortexChatResult:
        """
        Sends a CortexChatRequest to the Gateway and waits for a CortexChatResult.
        """
        correlation_id = str(uuid.uuid4())
        reply_to = f"{settings.CORTEX_GATEWAY_RESULT_PREFIX}:{correlation_id}"

        envelope = BaseEnvelope(
            kind="cortex.gateway.chat.request",  # Must match gateway subscription
            source=self._source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=request.model_dump(),
        )

        logger.info(f"[{correlation_id}] Sending chat request to {settings.CORTEX_GATEWAY_REQUEST_CHANNEL}")

        try:
            msg = await self.bus.rpc_request(
                settings.CORTEX_GATEWAY_REQUEST_CHANNEL,
                envelope,
                reply_channel=reply_to,
                timeout_sec=settings.TIMEOUT_SEC,
            )
        except TimeoutError:
             logger.error(f"[{correlation_id}] Request timed out.")
             raise

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
             logger.error(f"[{correlation_id}] Decode failed: {decoded.error}")
             raise ValueError(f"Bus decode error: {decoded.error}")

        # Strict validation of response
        payload = decoded.envelope.payload

        # If the gateway returns a dict (likely), parse it into the model
        if isinstance(payload, dict):
            try:
                return CortexChatResult.model_validate(payload)
            except Exception as e:
                logger.error(f"[{correlation_id}] Response validation failed: {e}")
                raise ValueError(f"Invalid response format from Gateway: {e}")
        elif isinstance(payload, CortexChatResult):
             return payload
        else:
             raise ValueError(f"Unexpected payload type: {type(payload)}")
