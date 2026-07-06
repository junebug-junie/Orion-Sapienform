from __future__ import annotations

import logging
import uuid
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.harness_finalize import HarnessRunRequestV1, HarnessRunV1
from scripts.settings import settings

logger = logging.getLogger("hub.bus.harness_governor")


class HarnessGovernorClient:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self._source = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

    async def run(
        self,
        request: HarnessRunRequestV1,
        *,
        correlation_id: Optional[str] = None,
        timeout_sec: float | None = None,
    ) -> HarnessRunV1 | None:
        correlation_id = correlation_id or request.correlation_id or str(uuid.uuid4())
        reply_to = f"{settings.CHANNEL_HARNESS_RESULT_PREFIX}{correlation_id}"
        wait_sec = max(
            0.1,
            float(
                timeout_sec
                if timeout_sec is not None
                else settings.HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC
            ),
        )
        envelope = BaseEnvelope(
            kind="harness.run.request.v1",
            source=self._source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=request.model_dump(mode="json"),
        )
        try:
            msg = await self.bus.rpc_request(
                settings.CHANNEL_HARNESS_RUN_REQUEST,
                envelope,
                reply_channel=reply_to,
                timeout_sec=wait_sec,
            )
        except TimeoutError:
            logger.warning("[%s] harness governor RPC timeout", correlation_id)
            return None
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return None
        payload = decoded.envelope.payload
        if isinstance(payload, dict) and payload.get("error"):
            logger.warning(
                "[%s] harness governor RPC error payload=%s",
                correlation_id,
                payload.get("error"),
            )
            return None
        if isinstance(payload, dict):
            return HarnessRunV1.model_validate(payload)
        return None
