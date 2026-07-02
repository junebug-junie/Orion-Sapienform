"""Read-only LLM gateway RPC for context-exec RLM subcalls."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ChatResponsePayload, LLMMessage, ServiceRef

from .settings import settings

logger = logging.getLogger("orion-context-exec.llm_tools")


def _source() -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version)


def _corr_uuid(correlation_id: str | None) -> str:
    raw = str(correlation_id or "").strip()
    if raw:
        try:
            return str(uuid.UUID(raw))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))
    return str(uuid.uuid4())


async def llm_chat_route(
    bus: OrionBusAsync | None,
    *,
    prompt: str,
    route: str,
    correlation_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    context: Any = None,
    schema: str | None = None,
    messages: list[LLMMessage] | None = None,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    """Invoke LLM gateway bus RPC with an explicit trusted route id."""
    route_key = str(route or "").strip().lower()
    if not settings.orion_bus_enabled or bus is None:
        return {
            "ok": False,
            "route": route_key,
            "summary": "llm bus unavailable",
            "error": "bus_disabled",
        }

    reply_channel = f"orion:exec:result:LLMGatewayService:{uuid.uuid4().hex}"
    corr = _corr_uuid(correlation_id)
    chat_messages = messages if messages else [LLMMessage(role="user", content=prompt)]
    options: dict[str, Any] = {}
    if schema:
        options["schema"] = schema
    if stop:
        options["stop"] = list(stop)
    if context is not None:
        options["context"] = context
    req = ChatRequestPayload(
        messages=chat_messages,
        route=route_key,
        raw_user_text=prompt,
        session_id=session_id,
        user_id=user_id,
        options=options,
    )

    env = BaseEnvelope(
        kind="llm.chat.request",
        source=_source(),
        correlation_id=corr,
        reply_to=reply_channel,
        payload=req.model_dump(mode="json"),
    )
    started = time.perf_counter()
    try:
        msg = await bus.rpc_request(
            settings.channel_llm_intake,
            env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.context_exec_llm_timeout_sec),
        )
    except Exception as exc:
        logger.warning("llm_chat_route rpc failed route=%s corr=%s err=%s", route_key, corr, exc)
        return {
            "ok": False,
            "route": route_key,
            "summary": "llm rpc failed",
            "error": str(exc),
        }

    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        return {
            "ok": False,
            "route": route_key,
            "summary": "llm decode failed",
            "error": decoded.error or "decode_failed",
        }

    payload_data = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    try:
        reply = ChatResponsePayload.model_validate(payload_data)
    except Exception as exc:
        return {
            "ok": False,
            "route": route_key,
            "summary": "llm response invalid",
            "error": str(exc),
            "raw": payload_data,
        }

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "ok": True,
        "route": route_key,
        "content": reply.content,
        "summary": (reply.content or "")[:240],
        "elapsed_ms": elapsed_ms,
        "correlation_id": corr,
    }
