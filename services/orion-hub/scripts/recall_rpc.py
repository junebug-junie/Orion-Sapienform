"""scripts.recall_rpc

Async Recall RPC client.

This is a *client* (Hub) → (Recall service) interaction. Hub should not own any
Redis subscribe loops or thread bridges; those details are centralized in
`orion.core.bus.rpc_async`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from .settings import settings

from orion.core.bus.rpc_async import request_and_wait

logger = logging.getLogger("hub.recall-rpc")


class RecallRPC:
    """Bus-RPC client for the Orion Recall service."""

    def __init__(self, bus: Any):
        self.bus = bus
        self.request_channel = settings.CHANNEL_RECALL_REQUEST
        self.reply_prefix = settings.CHANNEL_RECALL_REPLY_PREFIX

        if not self.bus or not getattr(self.bus, "enabled", False):
            logger.warning("[RecallRPC] Orion bus is disabled; calls will fail.")

    async def call_recall(
        self,
        *,
        query: str,
        mode: str = "hybrid",
        time_window_days: int = 30,
        max_items: int = 16,
        extras: Optional[dict] = None,
        trace_id: Optional[str] = None,
        timeout_sec: float = 30.0,
    ) -> Dict[str, Any]:
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("RecallRPC used while Orion bus is disabled")

        trace_id = trace_id or str(uuid.uuid4())
        reply_channel = f"{self.reply_prefix}:{trace_id}"

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "source": settings.SERVICE_NAME,
            "reply_channel": reply_channel,
            "query": query,
            "mode": mode,
            "time_window_days": time_window_days,
            "max_items": max_items,
            "extras": extras or {},
        }

        logger.info(
            "[RecallRPC] -> %s (reply=%s, mode=%s, max_items=%s)",
            self.request_channel,
            reply_channel,
            mode,
            max_items,
        )

        raw_reply = await request_and_wait(
            self.bus,
            intake_channel=self.request_channel,
            reply_channel=reply_channel,
            payload=payload,
            timeout_sec=timeout_sec,
        )

        if isinstance(raw_reply, dict):
            return raw_reply
        return {"trace_id": trace_id, "raw": raw_reply}
