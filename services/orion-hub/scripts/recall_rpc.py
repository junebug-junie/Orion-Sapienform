from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .settings import settings

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import Envelope, RecallRequestPayload, ServiceRef

logger = logging.getLogger("hub.recall-rpc")


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        node=getattr(settings, "NODE_NAME", None),
        version=settings.SERVICE_VERSION,
    )


class RecallRPC:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus

    async def call_recall(
        self,
        *,
        text: str,
        max_items: int = 8,
        time_window_days: int = 90,
        mode: str = "hybrid",
        tags: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
        timeout_sec: float = 60.0,
    ) -> Dict[str, Any]:
        return await self.recall(
            text=text,
            max_items=max_items,
            time_window_days=time_window_days,
            mode=mode,
            tags=tags,
            trace_id=trace_id,
            timeout_sec=timeout_sec,
        )

    async def recall(
        self,
        *,
        text: str,
        max_items: int = 8,
        time_window_days: int = 90,
        mode: str = "hybrid",
        tags: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
        timeout_sec: float = 60.0,
    ) -> Dict[str, Any]:
        corr = uuid4()
        reply = f"{settings.CHANNEL_RECALL_REPLY_PREFIX}{corr}"

        req = Envelope[RecallRequestPayload](
            kind="recall.query.request",
            source=_source(),
            correlation_id=corr,
            reply_to=reply,
            payload=RecallRequestPayload(
                query_text=text,
                max_items=max_items,
                time_window_days=time_window_days,
                mode=mode,
                tags=tags or [],
                trace_id=trace_id,
            ),
        )

        msg = await self.bus.rpc_request(
            settings.CHANNEL_RECALL_REQUEST,
            req,
            reply_channel=reply,
            timeout_sec=timeout_sec,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return {"ok": False, "error": decoded.error}
        env = decoded.envelope
        return env.payload if isinstance(env.payload, dict) else env.model_dump()
