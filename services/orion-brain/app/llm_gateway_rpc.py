# services/orion-brain/app/llm_gateway_rpc.py
from __future__ import annotations

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import (
    CHANNEL_LLM_GATEWAY_EXEC_INTAKE,
    CHANNEL_LLM_GATEWAY_REPLY_PREFIX,
    SERVICE_NAME,
    LLM_MODEL,
)

logger = logging.getLogger("brain.llm-gateway-rpc")


async def _request_and_wait(
    bus,
    channel_intake: str,
    channel_reply: str,
    payload: Dict[str, Any],
    trace_id: str,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """
    Same robust pattern as Hub:
    - subscribe first
    - then publish
    - wait with timeout
    """
    sub = bus.raw_subscribe(channel_reply)

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def listener():
        try:
            for msg in sub:
                loop.call_soon_threadsafe(queue.put_nowait, msg)
                break
        finally:
            sub.close()

    asyncio.get_running_loop().run_in_executor(None, listener)

    bus.publish(channel_intake, payload)
    logger.info("[%s] LLM-GW RPC Published -> %s (awaiting %s)", trace_id, channel_intake, channel_reply)

    try:
        msg = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        reply = msg.get("data", {})
        logger.info("[%s] LLM-GW RPC reply received.", trace_id)
        return reply
    except asyncio.TimeoutError:
        logger.error("[%s] LLM-GW RPC timed out waiting for %s", trace_id, channel_reply)
        return {"error": "timeout"}


class LLMGatewayRPC:
    """
    Bus-RPC client for the LLM Gateway, from inside Brain.

    Used by BrainLLMService when handling Cortex exec_step messages.
    """

    def __init__(self, bus):
        self.bus = bus

    async def call_chat(
        self,
        prompt: str,
        history: List[Dict[str, Any]],
        temperature: float = 0.7,
        source: str = "brain-cortex",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("LLMGatewayRPC used while OrionBus is disabled")

        trace_id = str(uuid.uuid4())
        reply_channel = f"{CHANNEL_LLM_GATEWAY_REPLY_PREFIX}:{trace_id}"

        payload = {
            "event": "chat",
            "service": "LLMGatewayService",
            "correlation_id": trace_id,
            "reply_channel": reply_channel,
            "payload": {
                "source": source,
                "prompt": prompt,
                "history": history,
                "temperature": temperature,
                "model": LLM_MODEL,
                "session_id": session_id,
                "user_id": user_id,
                "ts": datetime.utcnow().isoformat(),
            },
        }

        raw = await _request_and_wait(
            self.bus,
            CHANNEL_LLM_GATEWAY_EXEC_INTAKE,
            reply_channel,
            payload,
            trace_id,
            timeout_s=90.0,
        )

        # Expecting what Hub sees: {"event":"chat_result","service":"LLMGatewayService",...,"payload":{"text":...}}
        if not isinstance(raw, dict):
            return {"text": "", "raw": raw}

        if "payload" in raw and isinstance(raw["payload"], dict):
            text = raw["payload"].get("text")
        else:
            text = raw.get("text")

        return {
            "text": text,
            "raw": raw,
        }
