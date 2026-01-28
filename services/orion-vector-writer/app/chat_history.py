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
    if (envelope.payload.memory_status or "").lower() == "rejected":
        return None

    doc = envelope.payload.to_document(
        kind=envelope.kind,
        source=envelope.source,
        correlation_id=envelope.correlation_id,
        channel=channel,
    )
    metadata = dict(doc["metadata"])
    timestamp = metadata.get("timestamp")
    if isinstance(timestamp, str) and " " in timestamp and "T" not in timestamp:
        metadata["timestamp"] = timestamp.replace(" ", "T", 1)
    metadata["original_channel"] = channel
    metadata["correlation_id"] = str(envelope.correlation_id)
    metadata["envelope_id"] = str(envelope.id)
    created_at = envelope.created_at.isoformat()
    if " " in created_at and "T" not in created_at:
        created_at = created_at.replace(" ", "T", 1)
    metadata["created_at"] = created_at

    collection = collection_name or CHAT_HISTORY_COLLECTION
    return VectorWriteRequest(
        id=doc["id"],
        kind=envelope.kind,
        content=doc["text"],
        metadata=metadata,
        collection_name=collection,
    )
