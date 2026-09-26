"""One LLM gateway RPC for REM recombination.

Same bus pattern as orion-memory-consolidation/app/classify.py: publish an
`llm.chat.request` to the gateway intake, await one reply on a per-call channel.
Background lane, no chat fallback: a dream must never compete with a live turn.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

from app.settings import settings


async def complete(bus: Any, prompt: str) -> str:
    """Return the gateway's text. Raises on transport/decode failure."""
    rpc_corr = str(uuid4())
    reply_channel = f"orion:exec:result:LLMGatewayService:{rpc_corr}"
    route = settings.DREAM_LLM_ROUTE
    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)],
        route=route,
        options={
            "max_tokens": 320,
            "llm_route": route,
            "llm_lane": "background",
            "allow_chat_fallback": False,
            "purpose": "dream_recombine",
            "skip_spark_candidate_publish": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "gateway_read_timeout_sec": float(settings.DREAM_LLM_TIMEOUT_SEC),
        },
    )
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            node=settings.NODE_NAME,
        ),
        correlation_id=rpc_corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(
        settings.CHANNEL_LLM_INTAKE,
        env,
        reply_channel=reply_channel,
        timeout_sec=float(settings.DREAM_LLM_TIMEOUT_SEC),
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error)
    result = decoded.envelope.payload or {}
    return str(result.get("content") or result.get("text") or "")
