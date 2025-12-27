"""scripts.llm_rpc

Async RPC clients used by Hub.

Hub is a *bus client* (publish + wait). It should not own any Redis pubsub
listener loops or thread bridges.

All request/response waiting is centralized in `orion.core.bus.rpc_async`.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from .settings import settings
from orion.core.bus.rpc_async import request_and_wait

logger = logging.getLogger("hub.llm-rpc")


class CortexOrchRPC:
    """Bus-RPC client for the Orion Cortex Orchestrator."""

    def __init__(self, bus):
        self.bus = bus
        self.request_channel = getattr(settings, "CORTEX_ORCH_REQUEST_CHANNEL", "orion-cortex:request")
        self.result_prefix = getattr(settings, "CORTEX_ORCH_RESULT_PREFIX", "orion-cortex:result")

        if not self.bus or not getattr(self.bus, "enabled", False):
            logger.warning("[CortexOrchRPC] bus disabled; calls will fail")

    async def run_verb(
        self,
        *,
        verb_name: str,
        context: dict,
        steps: Optional[list[dict]] = None,
        origin_node: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict:
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("CortexOrchRPC used while OrionBus is disabled")

        trace_id = str(uuid.uuid4())
        result_channel = f"{self.result_prefix}:{trace_id}"

        core_payload: Dict[str, Any] = {
            "verb_name": verb_name,
            "origin_node": origin_node or settings.SERVICE_NAME,
            "context": context or {},
            "steps": steps or [],
            "timeout_ms": timeout_ms,
        }

        # Cortex Orch worker expects `trace_id` and (optionally) `result_channel`.
        # It also supports nested payloads (it will flatten), so we send both.
        envelope: Dict[str, Any] = {
            "trace_id": trace_id,
            "origin_node": origin_node or settings.SERVICE_NAME,
            "result_channel": result_channel,
            **core_payload,
            "payload": core_payload,
        }

        raw_reply = await request_and_wait(
            self.bus,
            intake_channel=self.request_channel,
            reply_channel=result_channel,
            payload=envelope,
            timeout_sec=(timeout_ms or 60000) / 1000.0,
        )

        # Some workers wrap data as {trace_id, ok, data={...}, error?}.
        # Merge so callers can reliably read `ok` and still access
        # `step_results` at the top level (Hub chat_front expects this).
        if isinstance(raw_reply, dict) and isinstance(raw_reply.get("data"), dict):
            merged = dict(raw_reply["data"])
            merged.setdefault("ok", raw_reply.get("ok"))
            merged.setdefault("trace_id", raw_reply.get("trace_id") or trace_id)
            if raw_reply.get("error") and "error" not in merged:
                merged["error"] = raw_reply.get("error")
            return merged

        # If it's already unwrapped, pass through.
        if isinstance(raw_reply, dict):
            raw_reply.setdefault("ok", True)
            raw_reply.setdefault("trace_id", trace_id)
            return raw_reply

        return {"ok": False, "trace_id": trace_id, "raw": raw_reply}

    async def run_chat_general(
        self,
        *,
        context: dict,
        origin_node: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict:
        return await self.run_verb(
            verb_name="chat_general",
            context=context,
            steps=[],
            origin_node=origin_node,
            timeout_ms=timeout_ms or 60000,
        )


class CouncilRPC:
    """Bus-RPC client for the Agent Council service."""

    def __init__(self, bus):
        self.bus = bus

    async def call_llm(self, prompt: str, history: list, temperature: float):
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("CouncilRPC used while OrionBus is disabled")

        trace_id = str(uuid.uuid4())
        reply_channel = f"{settings.CHANNEL_COUNCIL_REPLY_PREFIX}:{trace_id}"

        payload = {
            "event": "council_deliberation",
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "response_channel": reply_channel,
            "prompt": prompt,
            "history": (history or [])[-5:],
            "temperature": temperature,
            "mode": "council",
            "ts": datetime.utcnow().isoformat(),
        }

        raw_reply = await request_and_wait(
            self.bus,
            intake_channel=settings.CHANNEL_COUNCIL_INTAKE,
            reply_channel=reply_channel,
            payload=payload,
            timeout_sec=300.0,
        )

        if isinstance(raw_reply, dict):
            if "final_text" in raw_reply and "text" not in raw_reply:
                return {**raw_reply, "text": raw_reply["final_text"]}
            return raw_reply

        return {"raw": raw_reply}


class AgentChainRPC:
    """Bus-RPC client for the Orion Agent Chain service."""

    def __init__(self, bus):
        self.bus = bus
        self.request_channel = settings.CHANNEL_AGENT_CHAIN_INTAKE
        self.result_prefix = settings.CHANNEL_AGENT_CHAIN_REPLY_PREFIX

        if not self.bus or not getattr(self.bus, "enabled", False):
            logger.warning("[AgentChainRPC] bus disabled; calls will fail")

    async def run(
        self,
        *,
        text: str,
        mode: str = "chat",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        trace_id: Optional[str] = None,
        timeout_sec: float = 1500.0,
    ) -> dict:
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("AgentChainRPC used while OrionBus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.result_prefix}:{trace_id}"

        payload: Dict[str, Any] = {
            "text": text,
            "mode": mode,
            "session_id": session_id,
            "user_id": user_id,
        }
        if messages:
            payload["messages"] = messages
        if tools:
            payload["tools"] = tools

        envelope = {
            "trace_id": trace_id,
            "reply_channel": reply_channel,
            "payload": payload,
        }

        raw_reply = await request_and_wait(
            self.bus,
            intake_channel=self.request_channel,
            reply_channel=reply_channel,
            payload=envelope,
            timeout_sec=timeout_sec,
        )

        if not isinstance(raw_reply, dict):
            return {"raw": raw_reply}

        # Agent-chain replies are wrapped: {"status": "ok", "data": {...}}
        data = raw_reply.get("data")
        status = raw_reply.get("status")

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                pass

        if isinstance(data, dict):
            out = dict(data)
            out["_envelope_status"] = status
            return out

        # Fallback: sometimes `text` is top-level
        if "text" in raw_reply:
            return raw_reply

        return raw_reply


class LLMGatewayRPC:
    """Bus-RPC client for the Orion LLM Gateway."""

    def __init__(self, bus):
        self.bus = bus
        self.service_name = getattr(settings, "LLM_GATEWAY_SERVICE_NAME", None) or "LLMGatewayService"
        self.exec_request_prefix = getattr(settings, "EXEC_REQUEST_PREFIX", "orion-exec:request")
        self.reply_prefix = getattr(settings, "CHANNEL_LLM_REPLY_PREFIX", "orion:llm:reply")

        if not self.bus or not getattr(self.bus, "enabled", False):
            logger.warning("[LLMGatewayRPC] bus disabled; LLM calls will fail")

    async def call_chat(
        self,
        *,
        prompt: str,
        history: list,
        temperature: float,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        source: str = "hub",
        verb: str | None = None,
        profile_name: str | None = None,
    ) -> dict:
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("LLMGatewayRPC used while OrionBus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.reply_prefix}:{trace_id}"

        messages = list(history or [])
        if prompt:
            prompt = str(prompt)
            if (
                not messages
                or (messages[-1].get("role") or "").lower() != "user"
                or (messages[-1].get("content") or "") != prompt
            ):
                messages.append({"role": "user", "content": prompt})

        cleaned: list[dict] = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            content = m.get("content", "")
            if role not in ("system", "user", "assistant"):
                role = "assistant" if role in ("orion", "bot", "assistant_orion") else "user"
            cleaned.append({"role": role, "content": content})
        messages = cleaned

        body = {
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False,
            "return_json": False,
            "trace_id": trace_id,
            "user_id": user_id,
            "session_id": session_id,
            "source": source,
            "verb": verb,
            "profile_name": profile_name,
        }

        envelope = {
            "event": "chat",
            "service": self.service_name,
            "correlation_id": trace_id,
            "reply_channel": reply_channel,
            "payload": {"body": body},
        }

        exec_channel = f"{self.exec_request_prefix}:{self.service_name}"

        raw_reply = await request_and_wait(
            self.bus,
            intake_channel=exec_channel,
            reply_channel=reply_channel,
            payload=envelope,
            timeout_sec=300.0,
        )

        payload_dict = raw_reply.get("payload") if isinstance(raw_reply, dict) else None
        text = ""
        if isinstance(payload_dict, dict):
            text = (payload_dict.get("text") or "").strip()

        return {"text": text, "raw": raw_reply}

    async def call_generate(
        self,
        *,
        prompt: str,
        temperature: float,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        source: str = "hub",
        verb: str | None = None,
        profile_name: str | None = None,
    ) -> dict:
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("LLMGatewayRPC used while OrionBus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.reply_prefix}:{trace_id}"

        body = {
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": False,
            "return_json": False,
            "trace_id": trace_id,
            "user_id": user_id,
            "session_id": session_id,
            "source": source,
            "verb": verb,
            "profile_name": profile_name,
        }

        envelope = {
            "event": "generate",
            "service": self.service_name,
            "correlation_id": trace_id,
            "reply_channel": reply_channel,
            "payload": {"body": body},
        }

        exec_channel = f"{self.exec_request_prefix}:{self.service_name}"

        raw_reply = await request_and_wait(
            self.bus,
            intake_channel=exec_channel,
            reply_channel=reply_channel,
            payload=envelope,
            timeout_sec=300.0,
        )

        payload_dict = raw_reply.get("payload") if isinstance(raw_reply, dict) else None
        text = ""
        if isinstance(payload_dict, dict):
            text = (payload_dict.get("text") or "").strip()

        return {"text": text, "raw": raw_reply}
