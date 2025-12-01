# services/orion-hub/scripts/recall_rpc.py
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from scripts.settings import settings

logger = logging.getLogger("orion-hub.recall-rpc")


class RecallRPC:
    """
    Bus-based RPC client for the Recall service.

    Hub → publishes request on CHANNEL_RECALL_REQUEST
    Recall → replies on a per-request reply_channel
    """

    def __init__(self, bus):
        self.bus = bus

    async def query(
        self,
        text: str,
        max_items: int = 10,
        time_window_days: int = 30,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("RecallRPC used while OrionBus is disabled")

        trace_id = str(uuid.uuid4())
        reply_channel = f"orion:recall:rpc:{trace_id}"

        payload = {
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "reply_channel": reply_channel,
            "query": text,
            "max_items": max_items,
            "time_window_days": time_window_days,
            "mode": mode,
            "ts": datetime.utcnow().isoformat(),
        }

        # Publish request
        self.bus.publish(settings.CHANNEL_RECALL_REQUEST, payload)
        logger.info(
            "[RecallRPC %s] Published recall request to %s",
            trace_id,
            settings.CHANNEL_RECALL_REQUEST,
        )

        # Wait for reply on dedicated channel
        sub = self.bus.raw_subscribe(reply_channel)
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
        data = msg.get("data", {}) or {}

        logger.info("[RecallRPC %s] Reply received.", trace_id)
        return data
