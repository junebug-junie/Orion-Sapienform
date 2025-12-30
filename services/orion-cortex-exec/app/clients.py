from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import ValidationError

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import (
    BaseEnvelope,
    ChatRequestPayload,
    ChatResponsePayload,
    RecallRequestPayload,
    RecallResultPayload,
    ServiceRef,
)
from orion.core.bus.contracts import KINDS
from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult, PlannerRequest, PlannerResponse
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.clients")


class _BaseClient:
    def __init__(self, bus: OrionBusAsync, channel: str, timeout_ms: int):
        self.bus = bus
        self.channel = channel
        self.timeout_sec = float(timeout_ms) / 1000.0

    async def _rpc(self, env: BaseEnvelope, *, reply_channel: str, timeout_sec: Optional[float] = None) -> dict:
        rpc_timeout = timeout_sec if timeout_sec is not None else self.timeout_sec
        msg = await self.bus.rpc_request(self.channel, env, reply_channel=reply_channel, timeout_sec=rpc_timeout)
        return msg


class LLMGatewayClient(_BaseClient):
    """
    Strict, typed client for the LLM Gateway.
    Prevents 'dict soup' by enforcing Pydantic models on both ends.
    """
    def __init__(self, bus: OrionBusAsync):
        super().__init__(bus, settings.channel_llm_intake, settings.step_timeout_ms)

    async def chat(
        self,
        *,
        source: ServiceRef,
        req: ChatRequestPayload,
        correlation_id: str,
        reply_to: str,
        timeout_sec: Optional[float] = None,
    ) -> ChatResponsePayload:
        env = BaseEnvelope(
            kind=KINDS.llm_chat_request,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )

        msg = await self._rpc(env, reply_channel=reply_to, timeout_sec=timeout_sec)
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        return ChatResponsePayload.model_validate(decoded.envelope.payload)


class RecallClient(_BaseClient):
    def __init__(self, bus: OrionBusAsync):
        super().__init__(bus, settings.channel_recall_intake, settings.step_timeout_ms)

    async def query(
        self,
        *,
        source: ServiceRef,
        req: RecallRequestPayload,
        correlation_id: str,
        reply_to: str,
        timeout_sec: Optional[float] = None,
    ) -> RecallResultPayload:
        env = BaseEnvelope(
            kind=KINDS.recall_query_request,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )
        msg = await self._rpc(env, reply_channel=reply_to, timeout_sec=timeout_sec)
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        try:
            return RecallResultPayload.model_validate(decoded.envelope.payload)
        except ValidationError as ve:
            raise RuntimeError(f"Invalid recall payload: {ve}") from ve


class AgentChainClient(_BaseClient):
    def __init__(self, bus: OrionBusAsync):
        super().__init__(bus, settings.channel_agent_chain_intake, settings.step_timeout_ms)

    async def run(
        self,
        *,
        source: ServiceRef,
        req: AgentChainRequest,
        correlation_id: str,
        reply_to: str,
    ) -> AgentChainResult:
        env = BaseEnvelope(
            kind=KINDS.agent_chain_request,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )
        msg = await self._rpc(env, reply_channel=reply_to)
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            return AgentChainResult(**payload)
        raise RuntimeError("AgentChain returned non-dict payload")


class PlannerReactClient(_BaseClient):
    def __init__(self, bus: OrionBusAsync):
        super().__init__(bus, settings.channel_planner_intake, settings.step_timeout_ms)

    async def run(
        self,
        *,
        source: ServiceRef,
        req: PlannerRequest,
        correlation_id: str,
        reply_to: str,
    ) -> PlannerResponse:
        env = BaseEnvelope(
            kind=KINDS.planner_request,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )
        msg = await self._rpc(env, reply_channel=reply_to)
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            return PlannerResponse(**payload)
        raise RuntimeError("PlannerReact returned non-dict payload")


class CouncilClient(_BaseClient):
    """
    Stubbed council worker to keep the pipeline contract intact.
    """
    def __init__(self, bus: OrionBusAsync):
        super().__init__(bus, settings.channel_council_intake, settings.step_timeout_ms)

    async def deliberate(
        self,
        *,
        source: ServiceRef,
        req: Dict[str, Any],
        correlation_id: str,
        reply_to: str,
    ) -> Dict[str, Any]:
        env = BaseEnvelope(
            kind=KINDS.council_request,
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req,
        )
        msg = await self._rpc(env, reply_channel=reply_to)
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        payload = decoded.envelope.payload
        if isinstance(payload, dict):
            return payload
        raise RuntimeError("Council returned non-dict payload")

