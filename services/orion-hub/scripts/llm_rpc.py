from __future__ import annotations

import uuid
import asyncio
import logging
from datetime import datetime

from scripts.settings import settings

logger = logging.getLogger("hub.llm-rpc")


async def _request_and_wait(
    bus,
    channel_intake: str,
    channel_reply: str,
    payload: dict,
    trace_id: str,
) -> dict:
    """
    Robust RPC helper: Subscribes FIRST, then publishes.

    This prevents the "Race Condition" where the service replies 
    before the Hub has finished setting up the subscription.
    """
    # 1. Open the subscription immediately
    sub = bus.raw_subscribe(channel_reply)

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

    # 3. Start listener in background executor
    asyncio.get_running_loop().run_in_executor(None, listener)

    # 4. NOW publish (while we are listening)
    bus.publish(channel_intake, payload)
    logger.info(
        "[%s] RPC Published -> %s (awaiting %s)",
        trace_id,
        channel_intake,
        channel_reply,
    )

    # 5. Wait for result with timeout
    try:
        msg = await asyncio.wait_for(queue.get(), timeout=60.0)
        reply = msg.get("data", {})
        logger.info("[%s] RPC reply received.", trace_id)
        return reply
    except asyncio.TimeoutError:
        logger.error("[%s] RPC timed out waiting for %s", trace_id, channel_reply)
        return {"error": "timeout"}
    finally:
        pass


# ─────────────────────────────────────────────
# Agent Council RPC
# ─────────────────────────────────────────────

class CouncilRPC:
    """
    Bus-RPC client for the Agent Council service.
    """

    def __init__(self, bus):
        self.bus = bus

    async def call_llm(self, prompt: str, history: list, temperature: float):
        """
        Hub → Agent Council over the bus.

        Council is responsible for picking its own LLM backend/model
        (which can be wired internally to LLM Gateway).
        """
        trace_id = str(uuid.uuid4())
        reply_channel = f"{settings.CHANNEL_COUNCIL_REPLY_PREFIX}:{trace_id}"

        payload = {
            "event": "council_deliberation",
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "response_channel": reply_channel,
            "prompt": prompt,
            "history": history[-5:],  # lightweight contextual tail
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

        # raw_reply is whatever the council publishes on the reply channel,
        # i.e., a CouncilResult dict.
        if isinstance(raw_reply, dict):
            result = raw_reply

            # Normalize shape for callers: ensure .text exists if only final_text is present.
            if "final_text" in result and "text" not in result:
                result = {**result, "text": result["final_text"]}

            return result

        return raw_reply


# ─────────────────────────────────────────────
# LLM Gateway RPC (Hub → LLMGatewayService)
# ─────────────────────────────────────────────

class LLMGatewayRPC:
    """
    Bus-RPC client for the Orion LLM Gateway (`LLMGatewayService`).

    This is the preferred path for generic Hub LLM calls.

    Wire format (Hub → Gateway):

      ExecutionEnvelope:
        {
          "event": "chat" | "generate",
          "service": "LLMGatewayService",
          "correlation_id": "<uuid>",
          "reply_channel": "orion:llm:reply:<uuid>",
          "payload": {
            "body": { ...ChatBody or GenerateBody... }
          }
        }

    Gateway reply (Gateway → Hub):

        {
          "event": "chat_result" | "generate_result",
          "service": "LLMGatewayService",
          "correlation_id": "<uuid>",
          "payload": {
            "text": "<LLM output>"
          }
        }

    NOTE:
      - Hub does NOT choose model or backend.
      - Routing is done entirely inside LLM Gateway via profiles + defaults.
    """

    def __init__(self, bus):
        self.bus = bus

        # These are surfaced from scripts.settings, with sane fallbacks.
        self.service_name = getattr(settings, "LLM_GATEWAY_SERVICE_NAME", None) or "LLMGatewayService"
        self.exec_request_prefix = getattr(settings, "EXEC_REQUEST_PREFIX", "orion-exec:request")
        self.reply_prefix = getattr(settings, "CHANNEL_LLM_REPLY_PREFIX", "orion:llm:reply")

        if not self.bus or not getattr(self.bus, "enabled", False):
            logger.warning(
                "[LLMGatewayRPC] OrionBus is disabled; LLM calls will fail."
            )

        logger.info(
            "[LLMGatewayRPC] Initialized (service=%s, exec_prefix=%s, reply_prefix=%s)",
            self.service_name,
            self.exec_request_prefix,
            self.reply_prefix,
        )

    # ---------- Public API ----------

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
        """
        High-level chat call:

          - history: prior messages, already in {"role", "content"} format
          - prompt: newest user wave (will be appended as a final user message)

        Hub does NOT choose model or backend anymore.
        Routing is done inside the LLM Gateway via profiles + defaults.
        """
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("LLMGatewayRPC used while OrionBus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.reply_prefix}:{trace_id}"

        # Build messages (history + latest user prompt)
        trimmed_history = history[-5:] if history else []
        messages = list(trimmed_history)
        if prompt:
            messages.append({"role": "user", "content": prompt})

        body = {
            "messages": messages,
            "options": {
                "temperature": temperature,
                # no backend override here – purely profile/settings-driven
            },
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
            "payload": {
                "body": body,
            },
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

        return {
            "text": text,
            "raw": raw_reply,
        }

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
        """
        Generate-style call (single prompt into the gateway).

        Hub does NOT choose model or backend anymore.
        Routing is done inside the LLM Gateway via profiles + defaults.
        """
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("LLMGatewayRPC used while OrionBus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.reply_prefix}:{trace_id}"

        body = {
            "prompt": prompt,
            "options": {
                "temperature": temperature,
            },
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
            "payload": {
                "body": body,
            },
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

        return {
            "text": text,
            "raw": raw_reply,
        }
