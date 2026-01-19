from __future__ import annotations

import logging
from typing import Optional, Any
from uuid import UUID

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vector.schemas import EmbeddingGenerateV1

from .settings import settings

logger = logging.getLogger("orion-llm-gateway")


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=getattr(settings, "node_name", None),
        version=settings.service_version,
    )


async def publish_assistant_embedding(
    bus: Any,
    *,
    text: str,
    doc_id: str,
    trace_id: Optional[UUID],
) -> None:
    if not text or not text.strip():
        return
    if not bus or not getattr(bus, "enabled", False):
        logger.warning("Embedding publish skipped doc_id=%s: bus unavailable.", doc_id)
        return

    request = EmbeddingGenerateV1(
        doc_id=doc_id,
        text=text,
        embedding_profile="default",
        include_latent=False,
    )
    envelope = BaseEnvelope(
        kind="embedding.generate.v1",
        source=_source(),
        correlation_id=trace_id,
        payload=request.model_dump(mode="json"),
    )

    try:
        await bus.publish(settings.channel_embedding_generate, envelope)
    except Exception as exc:
        logger.warning("Embedding publish failed doc_id=%s error=%s", doc_id, exc)
