from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .settings import settings
from orion.core.bus.service import OrionBus

logger = logging.getLogger("hub.llm-rpc")


async def _request_and_wait(
    bus,
    channel_intake: str,
    channel_reply: str,
    payload: dict,
    trace_id: str,
    timeout_sec: float = 1500.0,
) -> dict:
    """
    Robust RPC helper: Subscribes FIRST, then publishes.
    """

    # --- Normalize Cortex channels to actual bus names ---
    physical_intake = channel_intake
    physical_reply = channel_reply

    if channel_intake.startswith("orion:cortex:request"):
        physical_intake = channel_intake.replace(
            "orion:cortex:request", "orion-cortex:request", 1
        )

    if channel_reply.startswith("orion:cortex:result"):
        physical_reply = channel_reply.replace(
            "orion:cortex:result", "orion-cortex:result", 1
        )

    # 1. Open the subscription immediately
    sub = bus.raw_subscribe(physical_reply)

    # 2. Define the listener (consumer)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def listener():
        try:
            for msg in sub:
                loop.call_soon_threadsafe(queue.put_nowait, msg)
                break
        finally:
            sub.close()

    # 3. Start listener
    asyncio.get_running_loop().run_in_executor(None, listener)

    # 4. Publish
    bus.publish(physical_intake, payload)
    logger.info(
        "[%s] RPC Published -> %s (awaiting %s, timeout=%.1fs)",
        trace_id,
        physical_intake,
        physical_reply,
        timeout_sec,
    )

    # 5. Wait for result with DYNAMIC timeout
    try:
        msg = await asyncio.wait_for(queue.get(), timeout=timeout_sec)
        reply = msg.get("data", {})
        logger.info("[%s] RPC reply received on %s.", trace_id, physical_reply)
        return reply
    except asyncio.TimeoutError:
        logger.error("[%s] RPC timed out waiting for %s", trace_id, physical_reply)
        return {"error": "timeout"}
    finally:
        pass


class CortexOrchRPC:
    """
    Bus-RPC client for the Orion Cortex Orchestrator.
    """

    def __init__(self, bus):
        self.bus = bus
        self.request_channel = "orion-cortex:request"
        self.result_prefix = "orion-cortex:result"

        if not self.bus or not getattr(self.bus, "enabled", False):
            logger.warning("[CortexOrchRPC] OrionBus is disabled; calls will fail.")

        logger.info(
            "[CortexOrchRPC] Initialized (request_channel=%s, result_prefix=%s)",
            self.request_channel,
            self.result_prefix,
        )

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
        reply_channel = f"{self.result_prefix}:{trace_id}"

        core_payload: Dict[str, Any] = {
            "verb_name": verb_name,
            "origin_node": origin_node or settings.SERVICE_NAME,
            "context": context or {},
            "steps": steps or [],
            "timeout_ms": timeout_ms,
        }

        envelope: Dict[str, Any] = {
            "event": "orchestrate_verb",
            "trace_id": trace_id,
            "origin_node": origin_node or settings.SERVICE_NAME,
            "reply_channel": reply_channel,
            **core_payload,
            "payload": core_payload,
        }

        raw_reply = await _request_and_wait(
            self.bus,
            self.request_channel,
            reply_channel,
            envelope,
            trace_id,
            timeout_sec=(timeout_ms or 60000) / 1000.0,
        )

        return raw_reply.get("data") if isinstance(raw_reply, dict) and "data" in raw_reply else raw_reply

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
    """
    Bus-RPC client for the Agent Council service.
    """

    def __init__(self, bus):
        self.bus = bus

    async def call_llm(self, prompt: str, history: list, temperature: float):
        trace_id = str(uuid.uuid4())
        reply_channel = f"{settings.CHANNEL_COUNCIL_REPLY_PREFIX}:{trace_id}"

        payload = {
            "event": "council_deliberation",
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "response_channel": reply_channel,
            "prompt": prompt,
            "history": history[-5:],
            "temperature": temperature,
            "mode": "council",
            "ts": datetime.utcnow().isoformat(),
        }

        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("CouncilRPC used while OrionBus is disabled")

        raw_reply = await _request_and_wait(
            self.bus,
            settings.CHANNEL_COUNCIL_INTAKE,
            reply_channel,
            payload,
            trace_id,
        )

        if isinstance(raw_reply, dict):
            result = raw_reply
            if "final_text" in result and "text" not in result:
                result = {**result, "text": result["final_text"]}
            return result

        return raw_reply


# ─────────────────────────────────────────────
# Agents RPC (The Active Class)
# ─────────────────────────────────────────────

class AgentChainRPC:
    """
    Bus-RPC client for the Orion Agent Chain service.
    """

    def __init__(self, bus):
        self.bus = bus
        self.request_channel = settings.CHANNEL_AGENT_CHAIN_INTAKE
        self.result_prefix = settings.CHANNEL_AGENT_CHAIN_REPLY_PREFIX

        if not self.bus or not getattr(self.bus, "enabled", False):
            logger.warning("[AgentChainRPC] OrionBus is disabled; calls will fail.")

        logger.info(
            "[AgentChainRPC] Initialized (request_channel=%s, result_prefix=%s)",
            self.request_channel,
            self.result_prefix,
        )

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

        payload = {
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

        logger.info(f"[RPC-TRACE] Sending AgentChain request {trace_id}")

        raw_reply = await _request_and_wait(
            self.bus,
            self.request_channel,
            reply_channel,
            envelope,
            trace_id,
            timeout_sec=timeout_sec,
        )

        # DEBUG TRACING
        logger.info(f"[RPC-TRACE] {trace_id} <- Raw Reply Type: {type(raw_reply)}")
        if isinstance(raw_reply, dict):
            # Print keys to confirm if we have 'data' or 'text'
            logger.info(f"[RPC-TRACE] {trace_id} <- Raw Keys: {list(raw_reply.keys())}")
        else:
            logger.info(f"[RPC-TRACE] {trace_id} <- Raw Content: {str(raw_reply)[:200]}")

        if not isinstance(raw_reply, dict):
            return {"raw": raw_reply}

        # ─────────────────────────────────────────────────────────────
        # 🛠️ ROBUST UNWRAPPING LOGIC (The Fix)
        # ─────────────────────────────────────────────────────────────
        
        data = raw_reply.get("data")
        status = raw_reply.get("status")

        # DEBUG: Inspect the 'data' field inside the envelope
        logger.info(f"[RPC-TRACE] 'data' field type: {type(data)}")

        # 1. Double Serialization Check: If data is a string, decode it.
        if isinstance(data, str):
            logger.info("[RPC-TRACE] 'data' is string. Attempting json.loads...")
            try:
                data = json.loads(data)
                logger.info(f"[RPC-TRACE] json.loads success. New type: {type(data)}")
            except Exception as e:
                logger.warning(f"[RPC-TRACE] json.loads failed: {e}")
                pass

        # 2. If data is (now) a dict, that is our result.
        if isinstance(data, dict):
            has_text = "text" in data
            logger.info(f"[RPC-TRACE] Unwrap success. Has 'text': {has_text}")
            
            result = dict(data)
            result["_envelope_status"] = status 
            return result

        # 3. Fallback: If 'text' is in the top level (no wrapper), return raw_reply
        if "text" in raw_reply:
             logger.info("[RPC-TRACE] Found 'text' in top-level reply. Returning raw_reply.")
             return raw_reply

        # 4. Total Fallback
        logger.warning("[RPC-TRACE] Could not unwrap valid data. Returning raw.")
        return raw_reply


class LLMGatewayRPC:
    """
    Bus-RPC client for the Orion LLM Gateway.
    """

    def __init__(self, bus):
        self.bus = bus
        self.service_name = getattr(settings, "LLM_GATEWAY_SERVICE_NAME", None) or "LLMGatewayService"
        self.exec_request_prefix = getattr(settings, "EXEC_REQUEST_PREFIX", "orion-exec:request")
        self.reply_prefix = getattr(settings, "CHANNEL_LLM_REPLY_PREFIX", "orion:llm:reply")

        if not self.bus or not getattr(self.bus, "enabled", False):
            logger.warning("[LLMGatewayRPC] OrionBus is disabled; LLM calls will fail.")

        logger.info(
            "[LLMGatewayRPC] Initialized (service=%s, exec_prefix=%s, reply_prefix=%s)",
            self.service_name,
            self.exec_request_prefix,
            self.reply_prefix,
        )

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

        raw_reply = await _request_and_wait(
            self.bus,
            exec_channel,
            reply_channel,
            envelope,
            trace_id,
        )

        payload = raw_reply.get("payload") if isinstance(raw_reply, dict) else None
        text = ""
        if isinstance(payload, dict):
            text = (payload.get("text") or "").strip()

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

        raw_reply = await _request_and_wait(
            self.bus,
            exec_channel,
            reply_channel,
            envelope,
            trace_id,
        )

        payload = raw_reply.get("payload") if isinstance(raw_reply, dict) else None
        text = ""
        if isinstance(payload, dict):
            text = (payload.get("text") or "").strip()

        return {"text": text, "raw": raw_reply}
