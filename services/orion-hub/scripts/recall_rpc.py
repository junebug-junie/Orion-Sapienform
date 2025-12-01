# services/orion-hub/scripts/recall_rpc.py

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional

from .settings import settings

logger = logging.getLogger("hub.recall-rpc")


class RecallRPC:
    """
    Bus-based RPC client for the Recall service.

    Protocol (matches orion-recall/app/main.py):

      - Publish JSON payload to CHANNEL_RECALL_REQUEST with:
          - query / text / query_text
          - mode (optional)
          - max_items (optional)
          - time_window_days (optional)
          - tags (optional)
          - phi (optional dict)
          - trace_id
          - reply_channel (explicit)
          - session_id (optional, just passed through)

      - Recall service:
          - Listens on CHANNEL_RECALL_REQUEST (aioredis)
          - Builds RecallQuery from payload
          - Runs run_recall_pipeline(...)
          - Publishes JSON string to `reply_channel`
            containing:
              {
                "trace_id": ...,
                "fragments": [...],
                "debug": {...},
                ...
              }
    """

    def __init__(self, bus):
        self.bus = bus

    async def call_recall(
        self,
        query: Optional[str] = None,
        session_id: Optional[str] = None,
        mode: Optional[str] = None,
        time_window_days: Optional[int] = None,
        max_items: Optional[int] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a recall request over the Orion bus, wait for the reply,
        and return the decoded JSON body from the recall service.
        """
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("RecallRPC used while OrionBus is disabled or unavailable")

        trace_id = str(uuid.uuid4())
        reply_channel = (
            f"{settings.CHANNEL_RECALL_DEFAULT_REPLY_PREFIX}:{trace_id}"
        )

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "query": query or "",
            "mode": mode,
            "time_window_days": time_window_days,
            "max_items": max_items,
            "tags": [],
            "phi": None,
            "reply_channel": reply_channel,
            "session_id": session_id,
        }

        extras = extras or {}
        # Optional tags / phi hints
        if isinstance(extras.get("tags"), list):
            payload["tags"] = extras["tags"]
        if isinstance(extras.get("phi"), dict):
            payload["phi"] = extras["phi"]

        # Publish to recall request channel
        self.bus.publish(settings.CHANNEL_RECALL_REQUEST, payload)
        logger.info(
            "RecallRPC published request trace_id=%s → %s",
            trace_id,
            settings.CHANNEL_RECALL_REQUEST,
        )

        # Wait for reply on reply_channel
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

        raw_data = msg.get("data")

        # Recall service publishes JSON string via aioredis
        if isinstance(raw_data, (bytes, bytearray)):
            raw_data = raw_data.decode("utf-8", "ignore")

        if isinstance(raw_data, str):
            try:
                data = json.loads(raw_data)
            except Exception as e:
                logger.error("RecallRPC failed to json.loads reply: %s", e, exc_info=True)
                raise
        elif isinstance(raw_data, dict):
            # In case OrionBus ever decodes JSON for us
            data = raw_data
        else:
            logger.error(
                "RecallRPC got unexpected data type from bus: %r",
                type(raw_data),
            )
            raise RuntimeError("Unexpected recall reply payload type")

        logger.info(
            "RecallRPC received reply trace_id=%s (fragments=%d)",
            data.get("trace_id"),
            len((data.get("fragments") or [])),
        )

        return data
