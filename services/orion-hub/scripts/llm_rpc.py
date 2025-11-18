# scripts/llm_rpc.py
import uuid
import asyncio
import logging
from datetime import datetime
import json

from scripts.settings import settings

logger = logging.getLogger("hub.llm-rpc")


class BrainRPC:
    """
    A Redis-based request/response RPC client.
    Hub publishes LLM requests → waits for Brain replies on a unique channel.
    """

    def __init__(self, bus, kind: str | None = None):
        self.bus = bus
        self.kind = kind  # e.g. "warm_start" or None

    async def call_llm(self, prompt: str, history: list, temperature: float):
        """
        Publish an LLM request over the Orion bus and wait for the response.
        """
        trace_id = str(uuid.uuid4())
        reply_channel = f"orion:brain:rpc:{trace_id}"
        print('I added prompt:')
        print(prompt)
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

        # ✅ tag special calls
        if self.kind is not None:
            payload["kind"] = self.kind

        if not self.bus.enabled:
            raise RuntimeError("BrainRPC used while OrionBus is disabled")

        # DEBUG: log full outbound RPC payload
        try:
            debug_out = json.dumps(payload, indent=2)[:2000]
            logger.warning(f"[DEBUG][HUB → BrainRPC publish] Payload:\n{debug_out}")
        except Exception as e:
            logger.error(f"[DEBUG] Failed to serialize outbound payload: {e}")



        self.bus.publish(settings.CHANNEL_BRAIN_INTAKE, payload)
        logger.info(f"[{trace_id}] Published Brain RPC request to {settings.CHANNEL_BRAIN_INTAKE}")

        sub = self.bus.raw_subscribe(reply_channel)
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

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

        logger.warning(f"[DEBUG][BrainRPC ← Brain reply] Raw reply:\n{reply}")
        logger.info(f"[{trace_id}] Brain RPC reply received.")
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

        # Wait for reply
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
