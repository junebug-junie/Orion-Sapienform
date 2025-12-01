# services/orion-hub/scripts/recall_rpc.py
from __future__ import annotations

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from scripts.settings import settings
from .llm_rpc import _await_single_reply  # reuse helper

logger = logging.getLogger("hub.recall-rpc")


class RecallRPC:
    """
    Redis-based RPC client for the Recall service.

    Publishes recall requests to CHANNEL_RECALL_REQUEST, waits for a reply
    on a unique reply channel built from CHANNEL_RECALL_DEFAULT_REPLY_PREFIX.
    """

    def __init__(self, bus):
        self.bus = bus

    async def call_recall(
        self,
        query: Optional[str],
        session_id: Optional[str],
        mode: Optional[str] = None,
        time_window_days: Optional[int] = None,
        max_items: Optional[int] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("RecallRPC used while OrionBus is disabled")

        trace_id = str(uuid.uuid4())
        reply_channel = f"{settings.CHANNEL_RECALL_DEFAULT_REPLY_PREFIX}:{trace_id}"

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "response_channel": reply_channel,
            "ts": datetime.utcnow().isoformat(),
        }

        if query:
            payload["query"] = query

        payload["mode"] = mode or settings.RECALL_DEFAULT_MODE
        payload["time_window_days"] = (
            time_window_days or settings.RECALL_DEFAULT_TIME_WINDOW_DAYS
        )
        payload["max_items"] = max_items or settings.RECALL_DEFAULT_MAX_ITEMS

        if session_id:
            payload["session_id"] = session_id

        if extras:
            payload["extras"] = extras

        self.bus.publish(settings.CHANNEL_RECALL_REQUEST, payload)
        logger.info(
            "[%s] Published Recall RPC request → %s",
            trace_id,
            settings.CHANNEL_RECALL_REQUEST,
        )

        reply = await _await_single_reply(self.bus, reply_channel, trace_id)
        return reply or {}
