from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage, RecallDirective


async def suggest_once(
    bus: OrionBusAsync,
    *,
    transcript: str,
    cortex_request_channel: str,
    cortex_result_prefix: str,
    source: ServiceRef,
    timeout_sec: float = 120.0,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    trace_id = str(uuid.uuid4())
    reply_channel = f"{cortex_result_prefix}:{trace_id}"
    ctx = CortexClientContext(
        messages=[LLMMessage(role="user", content=transcript)],
        session_id=session_id,
        trace_id=trace_id,
        metadata={"transcript": transcript},
    )
    cortex_req = CortexClientRequest(
        mode="agent",
        verb="memory_graph_suggest",
        packs=[],
        options={},
        recall=RecallDirective(enabled=False),
        context=ctx,
    )
    envelope = BaseEnvelope(
        kind="cortex.orch.request",
        source=source,
        correlation_id=trace_id,
        reply_to=reply_channel,
        payload=cortex_req.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(
        cortex_request_channel,
        envelope,
        reply_channel=reply_channel,
        timeout_sec=timeout_sec,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"Cortex RPC decode failed: {decoded.error}")
    payload = decoded.envelope.payload
    if isinstance(payload, str):
        return json.loads(payload)
    return payload if isinstance(payload, dict) else payload.model_dump(mode="json")
