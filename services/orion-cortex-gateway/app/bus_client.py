import logging
from uuid import uuid4
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexClientRequest

from .settings import get_settings

logger = logging.getLogger(__name__)

class BusClient:
    def __init__(self):
        self.settings = get_settings()
        self.bus = OrionBusAsync(url=self.settings.orion_bus_url)
        self.reply_prefix = self.settings.channel_cortex_result_prefix

    async def connect(self):
        logger.info(f"Connecting to bus at {self.settings.orion_bus_url}")
        await self.bus.connect()

    async def close(self):
        logger.info("Closing bus connection")
        await self.bus.close()

    def _service_ref(self) -> ServiceRef:
        return ServiceRef(
            name=self.settings.service_name,
            version=self.settings.service_version,
            node=self.settings.node_name
        )

    async def rpc_call_cortex_orch(self, req: CortexClientRequest) -> Dict[str, Any]:
        corr = uuid4()
        reply_to = f"{self.reply_prefix}:{corr}"

        env = BaseEnvelope(
            kind="cortex.orch.request",
            source=self._service_ref(),
            correlation_id=corr,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )

        logger.info(
            f"RPC Request channel={self.settings.channel_cortex_request} "
            f"correlation_id={corr} reply_to={reply_to}"
        )

        try:
            msg = await self.bus.rpc_request(
                self.settings.channel_cortex_request,
                env,
                reply_channel=reply_to,
                timeout_sec=self.settings.gateway_rpc_timeout_sec
            )
        except TimeoutError as te:
            logger.error(f"RPC Timeout correlation_id={corr}")
            raise TimeoutError(f"RPC timed out after {self.settings.gateway_rpc_timeout_sec}s") from te

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            logger.error(f"Decode failed correlation_id={corr} error={decoded.error}")
            raise RuntimeError(f"Decode failed: {decoded.error}")

        payload = decoded.envelope.payload

        # Log minimal success
        logger.info(f"RPC Success correlation_id={corr} kind={decoded.envelope.kind}")

        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json")

        return payload  # Should be dict or primitive, or BaseEnvelope if unknown
