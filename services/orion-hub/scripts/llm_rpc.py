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

    This prevents the race where the service replies before Hub
    has finished setting up the subscription.

    NOTE (Hub-only shim):
    - Hub historically thinks in terms of `orion:cortex:*`
    - The Cortex-Orchestrator service is actually subscribed to `orion-cortex:*`
    We normalize that mapping *here* so we don't have to touch other services.
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

    # 1. Open the subscription immediately on the *physical* reply channel
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

    # 3. Start listener in background executor
    asyncio.get_running_loop().run_in_executor(None, listener)

    # 4. NOW publish (while we are listening) on the *physical* intake channel
    bus.publish(physical_intake, payload)
    logger.info(
        "[%s] RPC Published -> %s (awaiting %s)",
        trace_id,
        physical_intake,
        physical_reply,
    )

    # 5. Wait for result with timeout
    try:
        msg = await asyncio.wait_for(queue.get(), timeout=60.0)
        reply = msg.get("data", {})
        logger.info("[%s] RPC reply received on %s.", trace_id, physical_reply)
        return reply
    except asyncio.TimeoutError:
        logger.error("[%s] RPC timed out waiting for %s", trace_id, physical_reply)
        return {"error": "timeout"}
    finally:
        pass


# ─────────────────────────────────────────────
# Cortex Orchestrator RPC (Hub → Cortex-Orch)
# ─────────────────────────────────────────────

class CortexOrchRPC:
    """
    Bus-RPC client for the Orion Cortex Orchestrator.

    Hub uses this to run high-level "verbs" like `chat_general`
    instead of hand-rolling prompts and wiring directly to LLM Gateway.
    """

    def __init__(self, bus):
        self.bus = bus

        # Hub thinks in logical cortex channels; _request_and_wait maps to physical.
        self.request_channel = "orion:cortex:request"
        self.result_prefix = "orion:cortex:result"

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
        steps: list[dict],
        origin_node: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict:
        """
        Generic entrypoint for any Cortex verb.

        Mirrors the OrchestrateVerbRequest shape used by cortex-orch:

          {
            "verb_name": str,
            "origin_node": str,
            "context": dict,
            "steps": [CortexStepConfig-like dicts],
            "timeout_ms": int | None,
          }
        """
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("CortexOrchRPC used while OrionBus is disabled")

        trace_id = str(uuid.uuid4())
        reply_channel = f"{self.result_prefix}:{trace_id}"

        envelope = {
            "event": "orchestrate_verb",
            "trace_id": trace_id,
            "origin_node": origin_node or settings.SERVICE_NAME,
            "reply_channel": reply_channel,
            "payload": {
                "verb_name": verb_name,
                "origin_node": origin_node or settings.SERVICE_NAME,
                "context": context or {},
                "steps": steps or [],
                "timeout_ms": timeout_ms,
            },
        }

        raw_reply = await _request_and_wait(
            self.bus,
            self.request_channel,
            reply_channel,
            envelope,
            trace_id,
        )

        return raw_reply

    async def run_chat_general(
        self,
        *,
        context: dict,
        origin_node: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict:
        """
        Convenience wrapper for the `chat_general` verb.

        Hub doesn't need to remember the step wiring; we bake it here:

        - Single step: `llm_chat_general`
        - Service: `LLMGatewayService`
        - Prompt template: `chat_general.j2`
        """
        steps = [
            {
                "verb_name": "chat_general",
                "step_name": "llm_chat_general",
                "description": (
                    "Single-step, generalist chat response that interprets "
                    "intent/tone and generates a final reply."
                ),
                "order": 0,
                "services": ["LLMGatewayService"],
                "prompt_template": "chat_general.j2",
                "requires_gpu": True,
                "requires_memory": True,
            }
        ]

        return await self.run_verb(
            verb_name="chat_general",
            context=context,
            steps=steps,
            origin_node=origin_node,
            timeout_ms=timeout_ms,
        )


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

class AgentChainRPC:
    """
    Bus-RPC client for the Orion Agent Chain service.

    Hub uses this to send text + context and get back an AgentChainResult:
      { mode, text, structured, planner_raw }
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
    ) -> dict:
        """
        Mirrors AgentChainRequest in agent-chain/api.py:
          { text, mode, session_id, user_id, messages, tools }
        """
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

        raw_reply = await _request_and_wait(
            self.bus,
            self.request_channel,
            reply_channel,
            envelope,
            trace_id,
        )

        # raw_reply should be AgentChainResult.model_dump() from agent-chain
        # { mode, text, structured, planner_raw }
        return raw_reply if isinstance(raw_reply, dict) else {"raw": raw_reply}



# ─────────────────────────────────────────────
# LLM Gateway RPC (Hub → LLMGatewayService)
# ─────────────────────────────────────────────

class LLMGatewayRPC:
    """
    Bus-RPC client for the Orion LLM Gateway (`LLMGatewayService`).
    """

    def __init__(self, bus):
        self.bus = bus

        self.service_name = getattr(
            settings, "LLM_GATEWAY_SERVICE_NAME", None
        ) or "LLMGatewayService"
        self.exec_request_prefix = getattr(
            settings, "EXEC_REQUEST_PREFIX", "orion-exec:request"
        )
        self.reply_prefix = getattr(
            settings, "CHANNEL_LLM_REPLY_PREFIX", "orion:llm:reply"
        )

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

          - `history` is treated as the *full* message list (system + user + assistant)
            in OpenAI-style format.
          - `prompt` is the latest user wave, used for logging and *only* appended if
            history does not already end with that prompt as a user message.

        Hub owns context length / trimming. Gateway just forwards.
        """
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("LLMGatewayRPC used while OrionBus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.reply_prefix}:{trace_id}"

        # 1) Start from the full history as provided by Hub
        messages = list(history or [])

        # 2) Ensure the latest user prompt is present *once* at the end
        if prompt:
            prompt = str(prompt)
            if (
                not messages
                or (messages[-1].get("role") or "").lower() != "user"
                or (messages[-1].get("content") or "") != prompt
            ):
                messages.append({"role": "user", "content": prompt})

        # 3) Normalize roles so vLLM never sees weird ones like "orion"
        cleaned: list[dict] = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            content = m.get("content", "")

            # Map Orion-specific / odd roles to valid OpenAI roles
            if role not in ("system", "user", "assistant"):
                if role in ("orion", "bot", "assistant_orion"):
                    role = "assistant"
                else:
                    # Unknown roles become "user" as a safe default
                    role = "user"

            cleaned.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        messages = cleaned

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
        # unchanged from your current version
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
