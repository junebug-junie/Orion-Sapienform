from __future__ import annotations
import logging
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .settings import settings

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ChatRequestPayload, Envelope, BaseEnvelope, ServiceRef

logger = logging.getLogger("hub.rpc")


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        node=getattr(settings, "NODE_NAME", None),
        version=settings.SERVICE_VERSION,
    )


def _ensure_prefix(s: str) -> str:
    if not s:
        return ""
    return s if s.endswith(":") else f"{s}:"


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
        kind: str = "rpc.general.request",
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends a BaseEnvelope (payload is dict) via rpc_request.
        """
        corr = correlation_id or str(uuid4())
        reply_channel = f"{_ensure_prefix(reply_prefix)}{corr}"

        env = BaseEnvelope(
            kind=kind,
            source=_source(),
            correlation_id=corr,
            reply_to=reply_channel,
            payload=payload,
        )

        msg = await self.bus.rpc_request(
            intake,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return {"ok": False, "error": str(decoded.error)}

        resp_payload = decoded.envelope.payload
        if resp_payload is None:
            return {}

        if hasattr(resp_payload, "model_dump"):
            return resp_payload.model_dump()
        
        return resp_payload if isinstance(resp_payload, dict) else {"data": resp_payload}


class LLMGatewayRPC:
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
        if "content" in payload and "text" not in payload:
            payload["text"] = payload.get("content")
        payload.setdefault("ok", True)
        return payload


class CortexOrchRPC(_BaseRPC):
    def __init__(self, bus: OrionBusAsync):
        super().__init__(bus)

    async def run(
        self,
        *,
        verb_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        origin_node: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        timeout_sec: float = 300.0,
    ) -> Dict[str, Any]:
        if timeout_ms is not None:
            timeout_sec = float(timeout_ms) / 1000.0

        payload: Dict[str, Any] = {"verb_name": verb_name, "args": args}
        if context is not None:
            payload["context"] = context
        if origin_node is not None:
            payload["origin_node"] = origin_node

        resp = await self.request_and_wait(
            intake=settings.CORTEX_ORCH_REQUEST_CHANNEL,
            reply_prefix=_ensure_prefix(settings.CORTEX_ORCH_RESULT_PREFIX),
            payload=payload,
            timeout_sec=timeout_sec,
            kind="cortex.orch.request",
        )

        # [DEBUG] PRINT THE EXACT STRUCTURE
        try:
            logger.warning(f"DEBUG_HUB_RESP: {json.dumps(resp, default=str)}")
        except Exception:
            logger.warning(f"DEBUG_HUB_RESP (raw): {resp}")

        # ─────────────────────────────────────────────────────────────
        # Robust Unwrap Strategy
        # ─────────────────────────────────────────────────────────────
        
        # 1. Peel Layer 1 (Orchestrator Wrapper)
        l1 = resp.get("result", {})
        if not isinstance(l1, dict):
            l1 = {}

        # 2. Peel Layer 2 (Executor Wrapper)
        l2 = l1.get("result", {})
        if not isinstance(l2, dict):
            l2 = {}

        # 3. Find Steps (Plan Execution)
        # Note: Check for 'steps' (Executor) OR 'step_results' (Orchestrator legacy)
        steps = l2.get("steps") or l2.get("step_results") or []
        
        extracted_text = None

        if isinstance(steps, list):
            for step in reversed(steps):
                step_res_map = step.get("result", {})
                
                # Check LLMGatewayService
                llm_res = step_res_map.get("LLMGatewayService", {})
                
                candidate = (
                    llm_res.get("text") or 
                    llm_res.get("content") or 
                    llm_res.get("llm_output")
                )
                
                if candidate:
                    extracted_text = candidate
                    break
        
        if extracted_text:
            resp["text"] = extracted_text
        elif "content" in resp and "text" not in resp:
            resp["text"] = resp.get("content")

        return resp

    async def run_chat_general(
        self,
        *,
        messages: List[Dict[str, Any]],
        context: Dict[str, Any],
        origin_node: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        timeout_sec: float = 300.0,
    ) -> Dict[str, Any]:
        if timeout_ms is not None:
            timeout_sec = float(timeout_ms) / 1000.0

        if "messages" not in context:
            context = {**context, "messages": messages}

        return await self.run(
            verb_name="chat_general",
            args={},
            context=context,
            origin_node=origin_node,
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
            kind="agent.chain.request",
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
            kind="council.request",
        )
