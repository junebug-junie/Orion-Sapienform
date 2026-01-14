# services/orion-cortex-exec/app/clients.py
from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional

from pydantic import ValidationError
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import (
    BaseEnvelope,
    ChatRequestPayload,
    ChatResponsePayload,
    ServiceRef,
    RecallRequestPayload
)
from orion.core.contracts.recall import RecallQueryV1, RecallReplyV1
from orion.schemas.agents.schemas import (
    AgentChainRequest,
    AgentChainResult,
    PlannerRequest,
    PlannerResponse,
    DeliberationRequest,
    CouncilResult,
)
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
        logger.info(
            "RPC emit -> %s kind=%s corr=%s reply=%s timeout=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_to,
            rpc_timeout,
        )
        started = time.perf_counter()
        try:
            msg = await self.bus.rpc_request(
                self.channel,
                env,
                reply_channel=reply_to,
                timeout_sec=rpc_timeout  # <--- PASSED HERE
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.warning(
                "RPC error <- %s kind=%s corr=%s reply=%s elapsed=%.2fs error=%s",
                self.channel,
                env.kind,
                correlation_id,
                reply_to,
                elapsed,
                exc,
            )
            raise
        elapsed = time.perf_counter() - started
        logger.info(
            "RPC ok <- %s kind=%s corr=%s reply=%s elapsed=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_to,
            elapsed,
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
        req: RecallQueryV1,
        correlation_id: str,
        reply_to: str,
        timeout_sec: float,
    ) -> RecallReplyV1:

        payload_model = RecallRequestPayload(
            text=req.fragment,       # Map internal 'fragment' to bus 'text'
            session_id=req.session_id,
            # 'verb' and 'intent' are stripped as they are not part of the public recall schema
            # User defaults to None if not provided
        )

        env = BaseEnvelope(
            kind="recall.query.v1",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=payload_model.model_dump(mode="json", by_alias=True), # Sends {"text": "...", ...}
        )

        logger.info(
            "RPC emit -> %s kind=%s corr=%s reply=%s timeout=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_to,
            timeout_sec,
        )
        started = time.perf_counter()
        try:
            msg = await self.bus.rpc_request(
                self.channel,
                env,
                reply_channel=reply_to,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.warning(
                "RPC error <- %s kind=%s corr=%s reply=%s elapsed=%.2fs error=%s",
                self.channel,
                env.kind,
                correlation_id,
                reply_to,
                elapsed,
                exc,
            )
            raise
        elapsed = time.perf_counter() - started
        logger.info(
            "RPC ok <- %s kind=%s corr=%s reply=%s elapsed=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_to,
            elapsed,
        )

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"RecallService decode failed: {decoded.error}")

        # Note: If the service returns a standard payload, validation typically happens here.
        # Assuming RecallReplyV1 aligns with the response or needs similar adaptation.
        payload_data = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        if payload_data.get("error"):
            details = payload_data.get("details")
            detail_suffix = f" ({details})" if details else ""
            raise RuntimeError(f"RecallService error: {payload_data['error']}{detail_suffix}")
        try:
            return RecallReplyV1.model_validate(payload_data)
        except ValidationError as exc:
            raise RuntimeError(f"RecallService payload validation failed: {exc}") from exc


class PlannerReactClient:
    """Typed RPC client for PlannerReactService."""

    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.channel = settings.channel_planner_intake
        self.timeout = float(settings.step_timeout_ms) / 1000.0

    async def plan(
        self,
        source: ServiceRef,
        req: PlannerRequest,
        correlation_id: str,
        reply_to: str,
        timeout_sec: Optional[float] = None,
    ) -> PlannerResponse:
        env = BaseEnvelope(
            kind="agent.planner.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )
        rpc_timeout = timeout_sec or self.timeout
        logger.info(
            "RPC emit -> %s kind=%s corr=%s reply=%s timeout=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_to,
            rpc_timeout,
        )
        started = time.perf_counter()
        try:
            msg = await self.bus.rpc_request(
                self.channel,
                env,
                reply_channel=reply_to,
                timeout_sec=rpc_timeout,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.warning(
                "RPC error <- %s kind=%s corr=%s reply=%s elapsed=%.2fs error=%s",
                self.channel,
                env.kind,
                correlation_id,
                reply_to,
                elapsed,
                exc,
            )
            raise
        elapsed = time.perf_counter() - started
        logger.info(
            "RPC ok <- %s kind=%s corr=%s reply=%s elapsed=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_to,
            elapsed,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        return PlannerResponse.model_validate(decoded.envelope.payload)



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
        rpc_timeout = timeout_sec or self.timeout
        logger.info(
            "RPC emit -> %s kind=%s corr=%s reply=%s timeout=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_to,
            rpc_timeout,
        )
        started = time.perf_counter()
        try:
            msg = await self.bus.rpc_request(
                self.channel,
                env,
                reply_channel=reply_to,
                timeout_sec=rpc_timeout,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.warning(
                "RPC error <- %s kind=%s corr=%s reply=%s elapsed=%.2fs error=%s",
                self.channel,
                env.kind,
                correlation_id,
                reply_to,
                elapsed,
                exc,
            )
            raise
        elapsed = time.perf_counter() - started
        logger.info(
            "RPC ok <- %s kind=%s corr=%s reply=%s elapsed=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_to,
            elapsed,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Decode failed: {decoded.error}")
        return AgentChainResult.model_validate(decoded.envelope.payload)


class CouncilClient:
    """Typed RPC client for Agent Council checkpoints."""

    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.channel = settings.channel_council_intake
        self.reply_prefix = settings.channel_council_reply_prefix
        self.timeout = float(settings.step_timeout_ms) / 1000.0

    async def deliberate(
        self,
        source: ServiceRef,
        req: DeliberationRequest,
        correlation_id: str,
        reply_to: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> CouncilResult:
        reply_channel = reply_to or f"{self.reply_prefix}:{correlation_id}"
        env = BaseEnvelope(
            kind="council.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=req.model_dump(mode="json"),
        )
        rpc_timeout = timeout_sec or self.timeout
        logger.info(
            "RPC emit -> %s kind=%s corr=%s reply=%s timeout=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_channel,
            rpc_timeout,
        )
        started = time.perf_counter()
        try:
            msg = await self.bus.rpc_request(
                self.channel,
                env,
                reply_channel=reply_channel,
                timeout_sec=rpc_timeout,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.warning(
                "RPC error <- %s kind=%s corr=%s reply=%s elapsed=%.2fs error=%s",
                self.channel,
                env.kind,
                correlation_id,
                reply_channel,
                elapsed,
                exc,
            )
            raise
        elapsed = time.perf_counter() - started
        logger.info(
            "RPC ok <- %s kind=%s corr=%s reply=%s elapsed=%.2fs",
            self.channel,
            env.kind,
            correlation_id,
            reply_channel,
            elapsed,
        )

        decoded = self.bus.codec.decode(msg.get("data"))
        payload: Dict[str, Any] = {}
        if decoded.ok:
            payload_obj = decoded.envelope.payload or decoded.envelope
            payload = payload_obj if isinstance(payload_obj, dict) else payload
        else:
            raw_data = msg.get("data")
            if isinstance(raw_data, dict):
                payload = raw_data
        try:
            return CouncilResult.model_validate(payload)
        except Exception as exc:
            raise RuntimeError(f"Council response invalid: {payload}") from exc
