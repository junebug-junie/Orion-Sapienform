# scripts/llm_rpc.py
from __future__ import annotations

import uuid
import asyncio
import logging
from datetime import datetime
import json

from scripts.settings import settings

logger = logging.getLogger("hub.llm-rpc")


async def _await_single_reply(bus, reply_channel: str, trace_id: str) -> dict:
    """
    Shared helper: subscribe once, read 1 message, close.
    """
    sub = bus.raw_subscribe(reply_channel)
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
    msg = await queue.get()
    reply = msg.get("data", {})
    logger.info("[%s] RPC reply received on %s.", trace_id, reply_channel)
    return reply


class BrainRPC:
    """
    A Redis-based request/response RPC client for the Brain service.
    Hub publishes LLM requests → waits for Brain replies on a unique channel.
    """

    def __init__(self, bus, kind: str | None = None):
        self.bus = bus
        self.kind = kind  # e.g. "warm_start" or None

    async def call_llm(self, prompt: str, history: list, temperature: float):
        trace_id = str(uuid.uuid4())
        reply_channel = f"orion:brain:rpc:{trace_id}"

        payload = {
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "response_channel": reply_channel,
            "prompt": prompt,
            "history": history[-5:],         # lightweight contextual tail
            "temperature": temperature,
            "model": settings.LLM_MODEL,
            "ts": datetime.utcnow().isoformat(),
        }

        # tag special calls
        if self.kind is not None:
            payload["kind"] = self.kind

        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("BrainRPC used while OrionBus is disabled")

        self.bus.publish(settings.CHANNEL_BRAIN_INTAKE, payload)
        logger.info(
            "[%s] Published Brain RPC request → %s",
            trace_id,
            settings.CHANNEL_BRAIN_INTAKE,
        )

        reply = await _await_single_reply(self.bus, reply_channel, trace_id)
        return reply

    async def call_tts(self, text: str, tts_q: asyncio.Queue):
        """
        Publishes a TTS RPC request and streams the Brain's GPU TTS reply.
        """
        rpc_id = str(uuid.uuid4())
        reply_channel = f"orion:tts:rpc:{rpc_id}"

        # Publish request
        self.bus.publish(
            settings.CHANNEL_TTS_INTAKE,
            {
                "rpc_id": rpc_id,
                "text": text,
                "source": settings.SERVICE_NAME,
            },
        )

        # Wait for reply (streamed)
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


class CouncilRPC:
    """
    Bus-RPC client for the Agent Council service.
    Same shape of prompt/history as BrainRPC, but publishes to the
    council intake channel and waits on a council reply channel.
    """

    def __init__(self, bus):
        self.bus = bus

    async def call_llm(self, prompt: str, history: list, temperature: float):
        trace_id = str(uuid.uuid4())
        reply_channel = f"{settings.CHANNEL_COUNCIL_REPLY_PREFIX}:{trace_id}"

        payload = {
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "response_channel": reply_channel,
            "prompt": prompt,
            "history": history[-5:],  # same lightweight tail
            "temperature": temperature,
            "model": settings.LLM_MODEL,
            "mode": "council",
            "ts": datetime.utcnow().isoformat(),
        }

        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("CouncilRPC used while OrionBus is disabled")

        self.bus.publish(settings.CHANNEL_COUNCIL_INTAKE, payload)
        logger.info(
            "[%s] Published Council RPC request → %s",
            trace_id,
            settings.CHANNEL_COUNCIL_INTAKE,
        )

        reply = await _await_single_reply(self.bus, reply_channel, trace_id)
        return reply
