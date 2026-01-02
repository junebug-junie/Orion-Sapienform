# services/orion-hub/scripts/bus_clients/cortex_client.py
from __future__ import annotations

import logging
from uuid import uuid4
from typing import Optional, Dict, Any, List

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexChatRequest, CortexClientResult

from ..settings import settings

logger = logging.getLogger("orion-hub.cortex-client")


class CortexClient:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.service_ref = ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
        )

    async def send_chat_request(
        self,
        prompt: str,
        mode: str = "brain",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        verb: Optional[str] = None,
        packs: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        recall: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Sends a typed CortexChatRequest to the Cortex Gateway and awaits the response.
        """
        correlation_id = str(uuid4())

        # Build Typed Request
        request_payload = CortexChatRequest(
            prompt=prompt,
            mode=mode,
            verb=verb,
            packs=packs,
            options=options,
            recall=recall,
            session_id=session_id,
            user_id=user_id,
            trace_id=correlation_id,
            metadata=metadata,
        )

        reply_to = f"{settings.CORTEX_GATEWAY_RESULT_PREFIX}:{correlation_id}"

        # Build Envelope
        envelope = BaseEnvelope(
            kind="cortex.gateway.request",
            source=self.service_ref,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=request_payload.model_dump(mode="json"),
        )

        logger.info(
            "[%s] Sending Chat Request to %s (mode=%s)",
            correlation_id,
            settings.CORTEX_GATEWAY_REQUEST_CHANNEL,
            mode,
        )

        try:
            # RPC call
            msg = await self.bus.rpc_request(
                channel=settings.CORTEX_GATEWAY_REQUEST_CHANNEL,
                envelope=envelope,
                reply_channel=reply_to,
                timeout_sec=timeout_sec,
            )

            # Decode response
            decoded = self.bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                raise ValueError(f"Failed to decode response: {decoded.error}")

            response_payload = decoded.envelope.payload

            # If it's a dict, try to normalize/validate if needed,
            # but for now we trust Gateway to return CortexClientResult-like dict
            if isinstance(response_payload, dict):
                return response_payload

            # Fallback if payload is something else
            return {"result": response_payload}

        except TimeoutError:
            logger.error("[%s] Chat Request timed out", correlation_id)
            return {"error": "Request timed out", "ok": False}
        except Exception as e:
            logger.error("[%s] Chat Request failed: %s", correlation_id, e, exc_info=True)
            return {"error": str(e), "ok": False}
