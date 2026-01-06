from __future__ import annotations

import logging
from typing import Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.chat_history import (
    CHAT_HISTORY_MESSAGE_KIND,
    ChatHistoryMessageEnvelope,
)
from orion.schemas.vector.schemas import VectorWriteRequest

logger = logging.getLogger(__name__)

CHAT_HISTORY_COLLECTION = "orion_chat"


def chat_history_envelope_to_request(
    env: BaseEnvelope,
    *,
    channel: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Optional[VectorWriteRequest]:
    """
    Validate and normalize a chat history envelope into a VectorWriteRequest.
    """
    if env.kind != CHAT_HISTORY_MESSAGE_KIND:
        return None

    try:
        # model_dump → json for UUID/string normalization
        envelope = ChatHistoryMessageEnvelope.model_validate(env.model_dump(mode="json"))
    except Exception as e:
        logger.error("Invalid chat history envelope: %s", e, exc_info=True)
        return None

    doc = envelope.payload.to_document(
        kind=envelope.kind,
        source=envelope.source,
        correlation_id=envelope.correlation_id,
        channel=channel,
    )

    collection = collection_name or CHAT_HISTORY_COLLECTION
    return VectorWriteRequest(
        id=doc["id"],
        kind=envelope.kind,
        content=doc["text"],
        metadata=doc["metadata"],
        collection_name=collection,
    )
