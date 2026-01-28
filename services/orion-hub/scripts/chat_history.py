from __future__ import annotations

import logging
from typing import Iterable, List, Optional
from uuid import UUID, uuid4

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.chat_history import (
    ChatHistoryMessageEnvelope,
    ChatHistoryMessageV1,
    ChatHistoryTurnEnvelope,
    ChatHistoryTurnV1,
    ChatRole,
)

from .settings import settings

logger = logging.getLogger("orion-hub.chat_history")


def build_chat_history_envelope(
    *,
    content: str,
    role: ChatRole,
    session_id: Optional[str],
    correlation_id: UUID | str | None,
    speaker: Optional[str],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tags: Optional[List[str]] = None,
    message_id: Optional[str] = None,
    memory_status: Optional[str] = None,
    memory_tier: Optional[str] = None,
    memory_reason: Optional[str] = None,
    client_meta: Optional[dict] = None,
) -> ChatHistoryMessageEnvelope:
    """
    Construct a versioned chat history envelope with Orion's canonical bus schema.
    """
    payload_kwargs = {
        "session_id": session_id,
        "role": role,
        "speaker": speaker,
        "content": content,
        "model": model,
        "provider": provider,
        "tags": tags or [],
        "memory_status": memory_status,
        "memory_tier": memory_tier,
        "memory_reason": memory_reason,
        "client_meta": client_meta,
    }
    if message_id:
        payload_kwargs["message_id"] = message_id

    payload = ChatHistoryMessageV1(**payload_kwargs)

    return ChatHistoryMessageEnvelope(
        correlation_id=correlation_id or uuid4(),
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            node=settings.NODE_NAME,
            version=settings.SERVICE_VERSION,
        ),
        payload=payload,
    )


async def publish_chat_history(
    bus, envelopes: Iterable[ChatHistoryMessageEnvelope]
) -> None:
    """
    Publish one or more chat history envelopes to the configured channel.
    """
    if not bus or not getattr(bus, "enabled", False):
        return
    if not settings.PUBLISH_CHAT_HISTORY_LOG:
        return

    channel = settings.chat_history_channel
    for env in envelopes:
        try:
            await bus.publish(channel, env)
        except Exception as e:
            logger.warning("Failed to publish chat history: %s", e, exc_info=True)

def build_chat_turn_envelope(
    *,
    prompt: str,
    response: str,
    session_id: Optional[str],
    correlation_id: UUID | str | None,
    user_id: Optional[str],
    source_label: str = "hub_ws",
    spark_meta: Optional[dict] = None,
    turn_id: Optional[str] = None,
    memory_status: Optional[str] = None,
    memory_tier: Optional[str] = None,
    memory_reason: Optional[str] = None,
    client_meta: Optional[dict] = None,
) -> ChatHistoryTurnEnvelope:
    """Construct a turn-level chat history envelope (prompt + response)."""
    merged_spark_meta = dict(spark_meta or {})
    if client_meta:
        merged_spark_meta.setdefault("client_meta", client_meta)
    payload = ChatHistoryTurnV1(
        id=turn_id,
        correlation_id=str(correlation_id) if correlation_id is not None else None,
        source=source_label,
        prompt=prompt,
        response=response,
        user_id=user_id,
        session_id=session_id,
        spark_meta=merged_spark_meta or None,
        memory_status=memory_status,
        memory_tier=memory_tier,
        memory_reason=memory_reason,
        client_meta=client_meta,
    )
    return ChatHistoryTurnEnvelope(
        correlation_id=correlation_id or uuid4(),
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            node=settings.NODE_NAME,
            version=settings.SERVICE_VERSION,
        ),
        payload=payload,
    )


async def publish_chat_turn(bus, env: ChatHistoryTurnEnvelope) -> None:
    """Publish a turn-level chat history envelope to the configured turn channel."""
    if not bus or not getattr(bus, "enabled", False):
        return
    if not settings.PUBLISH_CHAT_HISTORY_LOG:
        return

    channel = settings.chat_history_turn_channel
    try:
        await bus.publish(channel, env)
    except Exception as e:
        logger.warning("Failed to publish chat turn history: %s", e, exc_info=True)
