from __future__ import annotations

import logging
from typing import Optional
from uuid import uuid4

import httpx

from orion.core.bus.async_service import OrionBusAsync
from orion.memory.crystallization.bus_emit import emit_vector_upsert
from orion.memory.crystallization.projection_chroma import build_chroma_upsert
from orion.memory.crystallization.schemas import MemoryCrystallizationV1
from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1, VectorDocumentUpsertV1

logger = logging.getLogger(__name__)


async def _embed_http(
    *,
    text: str,
    doc_id: str,
    embed_host_url: str,
    timeout_ms: int,
    profile: str = "default",
) -> Optional[EmbeddingResultV1]:
    if not embed_host_url.strip():
        return None
    req = EmbeddingGenerateV1(doc_id=doc_id, text=text, embedding_profile=profile)
    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000.0) as client:
            resp = await client.post(embed_host_url.rstrip("/"), json=req.model_dump(mode="json"))
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                data.pop("latent", None)
                return EmbeddingResultV1.model_validate(data)
    except Exception as exc:
        logger.warning("crystallization_embed_http_failed doc_id=%s error=%s", doc_id, exc)
    return None


async def _embed_bus(
    bus: OrionBusAsync,
    *,
    text: str,
    doc_id: str,
    request_channel: str,
    result_channel_prefix: str,
    timeout_ms: int,
    profile: str = "default",
) -> Optional[EmbeddingResultV1]:
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    reply_channel = f"{result_channel_prefix}:{uuid4()}"
    env = BaseEnvelope(
        kind="embedding.generate.v1",
        source=ServiceRef(name="orion-hub", version="0.1.0", node="hub"),
        reply_to=reply_channel,
        payload=EmbeddingGenerateV1(doc_id=doc_id, text=text, embedding_profile=profile).model_dump(mode="json"),
    )
    try:
        msg = await bus.rpc_request(request_channel, env, reply_channel=reply_channel, timeout_sec=timeout_ms / 1000.0)
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            return None
        payload = decoded.envelope.payload
        payload_dict = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
        if isinstance(payload_dict, dict):
            payload_dict.pop("latent", None)
            return EmbeddingResultV1.model_validate(payload_dict)
    except Exception as exc:
        logger.warning("crystallization_embed_bus_failed doc_id=%s error=%s", doc_id, exc)
    return None


async def publish_crystallization_to_chroma(
    crystallization: MemoryCrystallizationV1,
    bus: Optional[OrionBusAsync],
    *,
    collection: str = "orion_memory_crystallizations",
    vector_channel: str = "orion:memory:vector:upsert",
    embed_host_url: str = "",
    embed_mode: str = "http",
    embed_request_channel: str = "orion:embedding:generate",
    embed_result_channel: str = "orion:embedding:result",
    embed_timeout_ms: int = 8000,
    service_name: str = "orion-hub",
) -> tuple[MemoryCrystallizationV1, dict]:
    """Build upsert, optionally embed, publish to vector bus. Returns updated crystallization + result."""
    result: dict = {"published": False, "skipped": False, "reason": None, "doc_id": None}

    upsert = build_chroma_upsert(crystallization, collection=collection)
    if upsert is None:
        result["skipped"] = True
        result["reason"] = "status_not_active"
        return crystallization, result

    embedding = None
    if embed_host_url.strip():
        embed = await _embed_http(
            text=upsert.text,
            doc_id=upsert.doc_id,
            embed_host_url=embed_host_url,
            timeout_ms=embed_timeout_ms,
        )
        if embed and embed.embedding:
            embedding = embed.embedding
            upsert = VectorDocumentUpsertV1(
                **{**upsert.model_dump(), "embedding": embedding, "embedding_model": embed.embedding_model, "embedding_dim": embed.embedding_dim or len(embedding)}
            )
    elif bus is not None and (embed_mode or "bus").lower() == "bus":
        embed = await _embed_bus(
            bus,
            text=upsert.text,
            doc_id=upsert.doc_id,
            request_channel=embed_request_channel,
            result_channel_prefix=embed_result_channel,
            timeout_ms=embed_timeout_ms,
        )
        if embed and embed.embedding:
            embedding = embed.embedding
            upsert = VectorDocumentUpsertV1(
                **{**upsert.model_dump(), "embedding": embedding, "embedding_model": embed.embedding_model, "embedding_dim": embed.embedding_dim or len(embedding)}
            )

    payload = upsert.model_dump(mode="json")
    if not payload.get("embedding"):
        result["skipped"] = True
        result["reason"] = "no_embedding"
        return crystallization, result

    published = await emit_vector_upsert(bus, payload=payload, channel=vector_channel, service_name=service_name)
    result["published"] = published
    result["doc_id"] = upsert.doc_id

    updated = crystallization.model_copy(deep=True)
    if upsert.doc_id not in updated.projection_refs.chroma_doc_ids:
        updated.projection_refs.chroma_doc_ids = list(updated.projection_refs.chroma_doc_ids) + [upsert.doc_id]
    from datetime import datetime, timezone

    updated.projection_refs.synced_at = datetime.now(timezone.utc)
    return updated, result
