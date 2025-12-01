from __future__ import annotations

import uuid
import asyncio
import logging
from datetime import datetime

from scripts.settings import settings

logger = logging.getLogger("hub.llm-rpc")


async def _request_and_wait(bus, channel_intake: str, channel_reply: str, payload: dict, trace_id: str) -> dict:
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
    logger.info("[%s] RPC Published -> %s (awaiting %s)", trace_id, channel_intake, channel_reply)

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


class BrainRPC:
    """
    A Redis-based request/response RPC client for the Brain service.
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
            trace_id
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
