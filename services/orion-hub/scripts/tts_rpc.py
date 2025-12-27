"""scripts.tts_rpc

Async TTS RPC client.

Hub publishes a request on CHANNEL_TTS_INTAKE and waits for a reply on
CHANNEL_TTS_RPC_PREFIX:<trace_id>.

All subscribe/wait boilerplate is handled by `orion.core.bus.rpc_async`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from .settings import settings

from orion.core.bus.rpc_async import request_and_wait

logger = logging.getLogger("hub.tts-rpc")


class TTSRPC:
    """Bus-RPC client for GPU TTS (currently implemented in orion-brain)."""

    def __init__(self, bus: Any):
        self.bus = bus

    async def call_tts(self, text: str, *, timeout_sec: float = 60.0) -> Dict[str, Any]:
        if not self.bus or not getattr(self.bus, "enabled", False):
            raise RuntimeError("TTSRPC used while Orion bus is disabled")

        trace_id = str(uuid.uuid4())
        response_channel = f"{settings.CHANNEL_TTS_RPC_PREFIX}:{trace_id}"

        payload = {
            "trace_id": trace_id,
            "text": text,
            "response_channel": response_channel,
            "source": settings.SERVICE_NAME,
        }

        logger.info(
            "[TTSRPC] -> %s (reply=%s)",
            settings.CHANNEL_TTS_INTAKE,
            response_channel,
        )

        raw_reply = await request_and_wait(
            self.bus,
            intake_channel=settings.CHANNEL_TTS_INTAKE,
            reply_channel=response_channel,
            payload=payload,
            timeout_sec=timeout_sec,
        )

        if isinstance(raw_reply, dict):
            return raw_reply
        return {"trace_id": trace_id, "raw": raw_reply}
