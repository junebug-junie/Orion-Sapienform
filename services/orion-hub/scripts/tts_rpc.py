# scripts/tts_rpc.py
import uuid
import asyncio
import logging
from datetime import datetime

from scripts.settings import settings

logger = logging.getLogger("hub.tts-rpc")


class TTSRPC:
    """
    Redis-based RPC client for TTS.
    Hub publishes text → Brain replies with base64 audio.
    """

    def __init__(self, bus):
        self.bus = bus

    async def call_tts(self, text: str):
        trace_id = str(uuid.uuid4())
        reply_channel = f"orion:tts:rpc:{trace_id}"

        payload = {
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "response_channel": reply_channel,
            "text": text,
            "ts": datetime.utcnow().isoformat(),
        }

        if not self.bus or not self.bus.enabled:
            raise RuntimeError("TTSRPC used while OrionBus is disabled or missing")

        # 1) Publish request
        self.bus.publish(settings.CHANNEL_TTS_INTAKE, payload)
        logger.info(f"[{trace_id}] Published TTS RPC request to {settings.CHANNEL_TTS_INTAKE}")

        # 2) Subscribe for a single reply (same pattern as BrainRPC)
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
        logger.info(f"[{trace_id}] TTS RPC reply received.")
        return reply
