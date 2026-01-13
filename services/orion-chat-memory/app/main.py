import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vector.schemas import (
    VectorDocumentUpsertV1,
    EmbeddingGenerateV1,
    EmbeddingResultV1,
)

from app.models import MemoryDocument
from app.settings import settings


logger = logging.getLogger(settings.SERVICE_NAME)

bus_hunter: Optional[Hunter] = None
http_client: Optional[httpx.AsyncClient] = None
chunk_counts: Dict[str, int] = {}


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        health_channel=settings.ORION_HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
    )


def _source() -> ServiceRef:
    return ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME)


def _should_emit_chunk(chunk_id: str, finalized: bool) -> bool:
    """
    Only embed chunk documents when finalized or every Nth message when configured.
    """
    interval = max(0, int(settings.CHAT_MEMORY_CHUNK_INTERVAL or 0))
    if finalized:
        chunk_counts.pop(chunk_id, None)
        return True

    if interval <= 0:
        # Only emit on finalization
        chunk_counts.setdefault(chunk_id, 0)
        return False

    next_count = chunk_counts.get(chunk_id, 0) + 1
    chunk_counts[chunk_id] = next_count
    return next_count % interval == 0


def _normalize_payload(env: BaseEnvelope) -> Optional[MemoryDocument]:
    payload: Dict[str, Any]
    if hasattr(env.payload, "model_dump"):
        payload = env.payload.model_dump()
    elif hasattr(env.payload, "dict"):
        payload = env.payload.dict()
    elif isinstance(env.payload, dict):
        payload = env.payload
    else:
        return None

    text = (
        payload.get("text")
        or payload.get("content")
        or payload.get("message")
        or ""
    )
    if not text or not str(text).strip():
        return None

    kind = payload.get("kind") or env.kind or "chat.message"
    doc_id = str(payload.get("id") or env.id)

    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("session_id", payload.get("session_id"))
    metadata.setdefault("user_id", payload.get("user_id"))
    metadata["source_channel"] = metadata.get("source_channel") or env.kind
    metadata["kind"] = kind
    metadata["created_at"] = str(env.created_at) if env.created_at else metadata.get("created_at")

    chunk_id = (
        payload.get("chunk_id")
        or payload.get("window_id")
        or (payload.get("chunk") or {}).get("id")
    )
    finalized = bool(
        payload.get("finalized")
        or payload.get("is_final")
        or payload.get("final")
        or payload.get("closed")
    )

    if chunk_id:
        metadata["chunk_id"] = chunk_id
        if not _should_emit_chunk(chunk_id, finalized):
            return None
        if finalized:
            doc_id = str(chunk_id)
        else:
            doc_id = f"{chunk_id}:{chunk_counts.get(chunk_id, 0)}"

    collection = payload.get("collection") or settings.CHAT_MEMORY_COLLECTION

    return MemoryDocument(
        doc_id=doc_id,
        kind=kind,
        text=str(text),
        metadata={k: v for k, v in metadata.items() if v is not None},
        collection=collection,
        chunk_id=chunk_id,
        finalized=finalized,
    )


async def _request_embedding_http(doc: MemoryDocument) -> Optional[EmbeddingResultV1]:
    if not settings.CHAT_MEMORY_EMBED_HOST_URL:
        logger.warning("HTTP embedding mode enabled but CHAT_MEMORY_EMBED_HOST_URL is not set.")
        return None
    if http_client is None:
        logger.warning("HTTP client not initialized; skipping embedding request.")
        return None

    req_payload = EmbeddingGenerateV1(
        doc_id=doc.doc_id,
        text=doc.text,
        embedding_profile=settings.CHAT_MEMORY_EMBED_PROFILE,
        include_latent=settings.CHAT_MEMORY_INCLUDE_LATENTS,
    ).model_dump(mode="json")

    try:
        resp = await http_client.post(
            settings.CHAT_MEMORY_EMBED_HOST_URL.rstrip("/"),
            json=req_payload,
            timeout=float(settings.CHAT_MEMORY_EMBED_TIMEOUT_MS) / 1000.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            logger.warning("Embedding host returned non-dict payload.")
            return None
        # Ensure we only retain lightweight latent data
        data.pop("latent", None)
        return EmbeddingResultV1.model_validate(data)
    except Exception as e:
        logger.warning(f"HTTP embedding request failed for {doc.doc_id}: {e}")
        return None


async def _request_embedding_bus(doc: MemoryDocument) -> Optional[EmbeddingResultV1]:
    if bus_hunter is None or bus_hunter.bus is None:
        logger.warning("Bus not initialized; cannot request embedding.")
        return None

    reply_channel = f"{settings.CHAT_MEMORY_EMBED_RESULT_CHANNEL}:{uuid4()}"
    env = BaseEnvelope(
        kind="embedding.generate.v1",
        source=_source(),
        reply_to=reply_channel,
        payload=EmbeddingGenerateV1(
            doc_id=doc.doc_id,
            text=doc.text,
            embedding_profile=settings.CHAT_MEMORY_EMBED_PROFILE,
            include_latent=settings.CHAT_MEMORY_INCLUDE_LATENTS,
        ).model_dump(mode="json"),
    )

    try:
        msg = await bus_hunter.bus.rpc_request(
            settings.CHAT_MEMORY_EMBED_REQUEST_CHANNEL,
            env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.CHAT_MEMORY_EMBED_TIMEOUT_MS) / 1000.0,
        )
        decoded = bus_hunter.bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            logger.warning(f"Embedding RPC decode failed: {decoded.error}")
            return None
        payload = decoded.envelope.payload
        payload_dict = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
        if isinstance(payload_dict, dict):
            payload_dict.pop("latent", None)
            return EmbeddingResultV1.model_validate(payload_dict)
    except Exception as e:
        logger.warning(f"Embedding RPC failed for {doc.doc_id}: {e}")
    return None


async def _fetch_embedding(doc: MemoryDocument) -> Optional[EmbeddingResultV1]:
    if not settings.CHAT_MEMORY_EMBED_ENABLE:
        return None

    mode = (settings.CHAT_MEMORY_EMBED_MODE or "bus").lower()
    if mode == "http":
        return await _request_embedding_http(doc)
    return await _request_embedding_bus(doc)


async def _publish_upsert(doc: MemoryDocument, embed: EmbeddingResultV1) -> None:
    if bus_hunter is None or bus_hunter.bus is None:
        logger.warning("Bus not initialized; cannot publish upsert.")
        return

    embedding = embed.embedding
    if not embedding:
        logger.warning(f"No embedding returned for {doc.doc_id}; skipping upsert.")
        return

    payload = VectorDocumentUpsertV1(
        doc_id=doc.doc_id,
        kind=doc.kind,
        text=doc.text,
        metadata=doc.metadata,
        collection=doc.collection,
        embedding=embedding,
        embedding_model=embed.embedding_model,
        embedding_dim=embed.embedding_dim or len(embedding),
        latent_ref=embed.latent_ref,
        latent_summary=embed.latent_summary,
    ).model_dump(mode="json")

    env = BaseEnvelope(
        kind="memory.vector.upsert.v1",
        source=_source(),
        payload=payload,
    )
    await bus_hunter.bus.publish(settings.CHAT_MEMORY_UPSERT_CHANNEL, env)
    logger.info(f"🧠 Published vector upsert for {doc.kind} (id={doc.doc_id})")


async def handle_envelope(env: BaseEnvelope) -> None:
    doc = _normalize_payload(env)
    if not doc:
        return

    embed = await _fetch_embedding(doc)
    if embed:
        await _publish_upsert(doc, embed)
    else:
        logger.warning(f"Embedding missing for {doc.doc_id}; not publishing upsert.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bus_hunter, http_client

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = _cfg()
    bus_hunter = Hunter(cfg, patterns=settings.SUBSCRIBE_CHANNELS, handler=handle_envelope)
    await bus_hunter.start_background()

    if (settings.CHAT_MEMORY_EMBED_MODE or "bus").lower() == "http":
        timeout = float(settings.CHAT_MEMORY_EMBED_TIMEOUT_MS) / 1000.0
        http_client = httpx.AsyncClient(timeout=timeout)

    yield

    if bus_hunter:
        await bus_hunter.stop()
    if http_client:
        await http_client.aclose()


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    bus_connected = bool(bus_hunter and bus_hunter.bus and getattr(bus_hunter.bus, "_redis", None))
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "bus_connected": bus_connected,
        "embed_mode": settings.CHAT_MEMORY_EMBED_MODE,
        "chunk_interval": settings.CHAT_MEMORY_CHUNK_INTERVAL,
    }
