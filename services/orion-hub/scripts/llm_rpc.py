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
        # standard timeout for the hub to give up
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
# Legacy Brain RPC (bus-based LLM + TTS)
# ─────────────────────────────────────────────

class BrainRPC:
    """
    A Redis-based request/response RPC client for the Brain service.

    NOTE:
      - `call_llm` is legacy and will be superseded by LLMGatewayRPC.
      - `call_tts` remains valid for GPU TTS routed through Brain.
    """

    def __init__(self, bus, kind: str | None = None):
        self.bus = bus
        self.kind = kind  # e.g. "warm_start" or None

    async def call_llm(self, prompt: str, history: list, temperature: float):
        """
        Legacy path: Hub → Brain LLM on CHANNEL_BRAIN_INTAKE.

        Prefer using LLMGatewayRPC for new call sites.
        """
        trace_id = str(uuid.uuid4())
        reply_channel = f"orion:brain:rpc:{trace_id}"

        payload = {
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "response_channel": reply_channel,
            "prompt": prompt,
            "history": history[-5:],  # lightweight contextual tail
            "temperature": temperature,
            "model": settings.LLM_MODEL,
            "ts": datetime.utcnow().isoformat(),
        }

        # tag special calls
        if self.kind is not None:
            payload["kind"] = self.kind

        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("BrainRPC used while OrionBus is disabled")

        return await _request_and_wait(
            self.bus,
            settings.CHANNEL_BRAIN_INTAKE,
            reply_channel,
            payload,
            trace_id,
        )

    async def call_tts(self, text: str, tts_q: asyncio.Queue):
        """
        Publishes a TTS RPC request and streams the Brain's GPU TTS reply.
        Note: TTS uses a stream, so we manually handle subscription here.
        """
        rpc_id = str(uuid.uuid4())
        reply_channel = f"orion:tts:rpc:{rpc_id}"

        # Note: Ideally we subscribe here before publishing too, 
        # but for streaming audio, a tiny race condition is less fatal 
        # than for control logic. Keeping as-is for now.

        self.bus.publish(
            settings.CHANNEL_TTS_INTAKE,
            {
                "rpc_id": rpc_id,
                "text": text,
                "source": settings.SERVICE_NAME,
            },
        )

        sub = self.bus.raw_subscribe(reply_channel)
        try:
            async for msg in sub:
                payload = msg.get("data", {})
                if payload.get("type") == "tts_chunk":
                    await tts_q.put({"audio_response": payload["chunk"]})
                if payload.get("type") == "tts_done":
                    break
        finally:
            sub.close()


# ─────────────────────────────────────────────
# Agent Council RPC (unchanged)
# ─────────────────────────────────────────────

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
            "model": settings.LLM_MODEL,
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
            # In your current wiring, council publishes the CouncilResult
            # directly as the message 'data', not nested in payload.
            result = raw_reply

            if "final_text" in result and "text" not in result:
                result = {**result, "text": result["final_text"]}

            return result

        return raw_reply


# ─────────────────────────────────────────────
# NEW: LLM Gateway RPC (Hub → LLMGatewayService)
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
    """

    def __init__(self, bus):
        self.bus = bus

        # These are surfaced from scripts.settings, with sane fallbacks.
        self.service_name = getattr(settings, "LLM_GATEWAY_SERVICE_NAME", None) or "LLMGatewayService"
        self.exec_request_prefix = getattr(settings, "EXEC_REQUEST_PREFIX", "orion-exec:request")
        self.reply_prefix = getattr(settings, "CHANNEL_LLM_REPLY_PREFIX", "orion:llm:reply")

        self.default_model = getattr(settings, "LLM_MODEL", "llama3.1:8b-instruct-q8_0")
        self.default_backend = getattr(settings, "LLM_BACKEND", "ollama")

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
        model: str | None = None,
        backend: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        source: str = "hub",
    ) -> dict:
        """
        High-level chat call:

          - history: prior messages, already in {"role", "content"} format
          - prompt: newest user wave (will be appended as a final user message)

        Returns:
          {
            "text": "<LLM output>",
            "raw":  { ...full gateway reply... }
          }
        """
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("LLMGatewayRPC used while OrionBus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.reply_prefix}:{trace_id}"

        # Build chat-style messages for the gateway.
        # We keep the last few history items and append the new user prompt.
        trimmed_history = history[-5:] if history else []
        messages = list(trimmed_history)
        if prompt:
            messages.append({"role": "user", "content": prompt})

        body = {
            "model": model or self.default_model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "backend": (backend or self.default_backend),
            },
            "stream": False,
            "return_json": False,
            "trace_id": trace_id,
            "user_id": user_id,
            "session_id": session_id,
            "source": source,
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

        # Normalize into a Hub-friendly shape.
        # Expected gateway reply:
        #   { event, service, correlation_id, payload: { text: "..." } }
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
        model: str | None = None,
        backend: str | None = None,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        source: str = "hub",
    ) -> dict:
        """
        Generate-style call (single prompt into the gateway).
        """
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("LLMGatewayRPC used while OrionBus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.reply_prefix}:{trace_id}"

        body = {
            "model": model or self.default_model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "backend": (backend or self.default_backend),
            },
            "stream": False,
            "return_json": False,
            "trace_id": trace_id,
            "user_id": user_id,
            "session_id": session_id,
            "source": source,
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
