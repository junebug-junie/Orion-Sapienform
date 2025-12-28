from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .settings import settings

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ChatRequestPayload, Envelope, ServiceRef

logger = logging.getLogger("hub.rpc")


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        node=getattr(settings, "NODE_NAME", None),
        version=settings.SERVICE_VERSION,
    )


class _BaseRPC:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus

    async def request_and_wait(
        self,
        *,
        intake: str,
        reply_prefix: str,
        payload: Dict[str, Any],
        timeout_sec: float,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        msg = dict(payload)
        if correlation_id:
            msg["correlation_id"] = correlation_id
        return await self.bus.rpc_legacy_dict(
            intake,
            msg,
            reply_prefix=reply_prefix,
            timeout_sec=timeout_sec,
        )


class LLMGatewayRPC:
    """
    RPC wrapper for orion-llm-gateway.
    Uses Titanium envelope on the wire.
    """

    def __init__(self, bus: OrionBusAsync):
        self.bus = bus

    async def call_chat(
        self,
        *,
        prompt: str,
        history: List[Dict[str, Any]],
        temperature: float = 0.7,
        source: str = "hub",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        profile: Optional[str] = None,
        timeout_sec: float = 1500.0,
    ) -> Dict[str, Any]:
        corr = uuid4()
        reply_channel = f"{settings.CHANNEL_LLM_REPLY_PREFIX}{corr}"

        messages = list(history or [])
        messages.append({"role": "user", "content": prompt})

        req = Envelope[ChatRequestPayload](
            kind="llm.chat.request",
            source=_source(),
            correlation_id=corr,
            reply_to=reply_channel,
            payload=ChatRequestPayload(
                model=model,
                profile=profile,
                messages=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
                options={"temperature": temperature},
                user_id=user_id,
                session_id=session_id,
            ),
        )

        msg = await self.bus.rpc_request(
            settings.CHANNEL_LLM_INTAKE,
            req,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return {"ok": False, "error": decoded.error}

        payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        # normalize to old keys used by hub
        if "content" in payload and "text" not in payload:
            payload["text"] = payload.get("content")
        payload.setdefault("ok", True)
        return payload


class CortexOrchRPC(_BaseRPC):
    def __init__(self, bus: OrionBusAsync):
        super().__init__(bus)

    async def run(self, *, verb_name: str, args: Dict[str, Any], timeout_sec: float = 300.0) -> Dict[str, Any]:
        return await self.request_and_wait(
            intake=settings.CORTEX_ORCH_REQUEST_CHANNEL,
            reply_prefix=settings.CORTEX_ORCH_RESULT_PREFIX,
            payload={"verb_name": verb_name, "args": args},
            timeout_sec=timeout_sec,
        )


class AgentChainRPC(_BaseRPC):
    def __init__(self, bus: OrionBusAsync):
        super().__init__(bus)

    async def run(self, *, text: str, mode: str, session_id: str, user_id: Optional[str] = None, timeout_sec: float = 900.0) -> Dict[str, Any]:
        return await self.request_and_wait(
            intake=settings.CHANNEL_AGENT_CHAIN_INTAKE,
            reply_prefix=settings.CHANNEL_AGENT_CHAIN_REPLY_PREFIX,
            payload={"text": text, "mode": mode, "session_id": session_id, "user_id": user_id},
            timeout_sec=timeout_sec,
        )


class CouncilRPC(_BaseRPC):
    def __init__(self, bus: OrionBusAsync):
        super().__init__(bus)

    async def ask(self, *, question: str, context: Dict[str, Any], timeout_sec: float = 900.0) -> Dict[str, Any]:
        return await self.request_and_wait(
            intake=settings.CHANNEL_COUNCIL_INTAKE,
            reply_prefix=settings.CHANNEL_COUNCIL_REPLY_PREFIX,
            payload={"question": question, "context": context},
            timeout_sec=timeout_sec,
        )
