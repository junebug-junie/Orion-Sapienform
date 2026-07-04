import logging
import uuid
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.pre_turn_appraisal import PreTurnAppraisalRequestV1, TurnAppraisalBundleV1
from scripts.settings import settings

logger = logging.getLogger("hub.bus.pre_turn_appraisal")


class PreTurnAppraisalClient:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self._source = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

    async def appraise(
        self,
        request: PreTurnAppraisalRequestV1,
        *,
        correlation_id: Optional[str] = None,
    ) -> TurnAppraisalBundleV1 | None:
        correlation_id = correlation_id or request.correlation_id or str(uuid.uuid4())
        reply_to = f"{settings.CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX}:{correlation_id}"
        timeout_sec = max(0.1, request.options.timeout_ms / 1000.0)
        envelope = BaseEnvelope(
            kind="pre_turn_appraisal.request.v1",
            source=self._source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=request.model_dump(mode="json"),
        )
        try:
            msg = await self.bus.rpc_request(
                settings.CHANNEL_PRE_TURN_APPRAISAL_REQUEST,
                envelope,
                reply_channel=reply_to,
                timeout_sec=timeout_sec,
            )
        except TimeoutError:
            logger.warning("[%s] pre_turn_appraisal RPC timeout", correlation_id)
            return None
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return None
        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            return TurnAppraisalBundleV1.model_validate(payload)
        return None
