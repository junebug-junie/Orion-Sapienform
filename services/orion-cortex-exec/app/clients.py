# services/orion-cortex-exec/app/clients.py
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
from orion.schemas.agents.schemas import AgentChainRequest, AgentChainResult
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
        reply_to: str,
        timeout_sec: Optional[float] = None,  # <--- ADDED ARGUMENT
    ) -> ChatResponsePayload:
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

        # Use passed timeout, or fall back to global default
        rpc_timeout = timeout_sec if timeout_sec is not None else self.timeout

        msg = await self.bus.rpc_request(
            self.channel,
            env,
            reply_channel=reply_to,
            timeout_sec=rpc_timeout  # <--- PASSED HERE
        )

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")

        return ChatResponsePayload.model_validate(decoded.envelope.payload)


class RecallClient:
    """Typed RPC client for recall queries."""

    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.channel = settings.channel_recall_intake

    async def query(
        self,
        source: ServiceRef,
        req: RecallRequestPayload,
        correlation_id: str,
        reply_to: str,
        timeout_sec: float,
    ) -> RecallResultPayload:
        env = BaseEnvelope(
            kind="recall.query.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )
        msg = await self.bus.rpc_request(
            self.channel, env, reply_channel=reply_to, timeout_sec=timeout_sec
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        return RecallResultPayload.model_validate(decoded.envelope.payload)


class AgentChainClient:
    """Typed RPC client for AgentChainService."""

    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.channel = settings.channel_agent_chain_intake
        self.timeout = float(settings.step_timeout_ms) / 1000.0

    async def run_chain(
        self,
        source: ServiceRef,
        req: AgentChainRequest,
        correlation_id: str,
        reply_to: str,
        timeout_sec: Optional[float] = None,
    ) -> AgentChainResult:
        env = BaseEnvelope(
            kind="agent.chain.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )
        msg = await self.bus.rpc_request(
            self.channel,
            env,
            reply_channel=reply_to,
            timeout_sec=timeout_sec or self.timeout,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        return AgentChainResult.model_validate(decoded.envelope.payload)
