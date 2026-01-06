from __future__ import annotations

import logging
from typing import Iterable, List, Optional
from uuid import UUID, uuid4

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.chat_history import ChatHistoryMessageEnvelope, ChatHistoryMessageV1, ChatRole

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
) -> ChatHistoryMessageEnvelope:
    """
    Construct a versioned chat history envelope with Orion's canonical bus schema.
    """
    payload = ChatHistoryMessageV1(
        message_id=message_id,
        session_id=session_id,
        role=role,
        speaker=speaker,
        content=content,
        model=model,
        provider=provider,
        tags=tags or [],
    )

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
