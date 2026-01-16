from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.chat_history import ChatHistoryMessageV1, CHAT_HISTORY_MESSAGE_KIND
from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1, VectorUpsertV1

from .embedder import Embedder
from .settings import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(settings.SERVICE_NAME)

embedder: Optional[Embedder] = None
hunter: Optional[Hunter] = None


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        health_channel=settings.ORION_HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
    )


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.NODE_NAME,
    )


def _base_meta(env: BaseEnvelope, *, original_channel: str, role: str, timestamp: str) -> Dict[str, Any]:
    return {
        "source_service": env.source.name,
        "source_node": env.source.node,
        "original_channel": original_channel,
        "role": role,
        "timestamp": timestamp,
        "correlation_id": str(env.correlation_id),
        "envelope_id": str(env.id),
        "vector_host_service": settings.SERVICE_NAME,
    }


async def _publish_semantic_upsert(
    *,
    env: BaseEnvelope,
    text: str,
    doc_id: str,
    role: str,
    meta: Dict[str, Any],
    embedding: list[float],
    embedding_model: str,
    embedding_dim: int,
    original_channel: str,
) -> None:
    if not hunter or not hunter.bus:
        logger.warning("Vector upsert skipped: bus unavailable.")
        return

    payload = VectorUpsertV1(
        doc_id=doc_id,
        collection=settings.VECTOR_HOST_SEMANTIC_COLLECTION,
        embedding=embedding,
        embedding_kind="semantic",
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        text=text,
        meta={k: v for k, v in meta.items() if v is not None},
    )
    envelope = BaseEnvelope(
        kind="vector.upsert.v1",
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=payload.model_dump(mode="json"),
    )
    await hunter.bus.publish(settings.VECTOR_HOST_SEMANTIC_UPSERT_CHANNEL, envelope)
    logger.info(
        "Semantic upsert published doc_id=%s role=%s channel=%s",
        doc_id,
        role,
        original_channel,
    )


async def _handle_chat_history(env: BaseEnvelope) -> None:
    if embedder is None:
        logger.warning("Embedding skipped: embedder unavailable.")
        return
    payload_obj = env.payload.model_dump(mode="json") if hasattr(env.payload, "model_dump") else env.payload
    if not isinstance(payload_obj, dict):
        return

    try:
        message = ChatHistoryMessageV1.model_validate(payload_obj)
    except Exception as exc:
        logger.warning("Chat history payload invalid: %s", exc)
        return

    role = message.role
    if role not in settings.EMBED_ROLES:
        return

    text = message.content
    if not text:
        return

    doc_id = message.message_id or str(env.correlation_id)
    timestamp = message.timestamp or env.created_at.isoformat()

    meta = _base_meta(
        env,
        original_channel=settings.VECTOR_HOST_CHAT_HISTORY_CHANNEL,
        role=role,
        timestamp=timestamp,
    )
    meta.update(
        {
            "message_id": message.message_id,
            "session_id": message.session_id,
            "speaker": message.speaker,
            "provider": message.provider,
            "model": message.model,
            "tags": message.tags,
        }
    )
    try:
        embedding, embedding_model, embedding_dim = await embedder.embed(text)
    except Exception as exc:
        logger.warning("Embedding failed doc_id=%s error=%s", doc_id, exc)
        return

    await _publish_semantic_upsert(
        env=env,
        text=text,
        doc_id=doc_id,
        role=role,
        meta=meta,
        embedding=embedding,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        original_channel=settings.VECTOR_HOST_CHAT_HISTORY_CHANNEL,
    )


def _extract_embedding_text(payload_obj: Dict[str, Any]) -> Optional[str]:
    text = payload_obj.get("text")
    if isinstance(text, str) and text.strip():
        return text
    raw_input = payload_obj.get("input")
    if isinstance(raw_input, list) and raw_input:
        return str(raw_input[0])
    if isinstance(raw_input, str):
        return raw_input
    return None


async def _handle_embedding_request(env: BaseEnvelope) -> None:
    if embedder is None or hunter is None:
        logger.warning("Embedding request skipped: service unavailable.")
        return

    payload_obj = env.payload.model_dump(mode="json") if hasattr(env.payload, "model_dump") else env.payload
    if not isinstance(payload_obj, dict):
        return

    try:
        request = EmbeddingGenerateV1.model_validate(payload_obj)
    except Exception as exc:
        logger.warning("Embedding request payload invalid: %s", exc)
        request = None

    text = _extract_embedding_text(payload_obj)
    doc_id = (
        (request.doc_id if request else None)
        or payload_obj.get("doc_id")
        or str(env.correlation_id)
    )

    reply_channel = env.reply_to or f"{settings.VECTOR_HOST_EMBEDDING_RESULT_PREFIX}{doc_id}"
    if not text:
        error_payload = EmbeddingResultV1(
            doc_id=doc_id,
            embedding=[],
            embedding_model=None,
            embedding_dim=0,
        ).model_dump(mode="json")
        error_payload["error"] = "missing_text"
        await hunter.bus.publish(
            reply_channel,
            BaseEnvelope(
                kind="embedding.result.v1",
                source=_source(),
                correlation_id=env.correlation_id,
                causality_chain=env.causality_chain,
                payload=error_payload,
            ),
        )
        return

    try:
        embedding, embedding_model, embedding_dim = await embedder.embed(text)
    except Exception as exc:
        logger.warning("Embedding request failed doc_id=%s error=%s", doc_id, exc)
        error_payload = EmbeddingResultV1(
            doc_id=doc_id,
            embedding=[],
            embedding_model=None,
            embedding_dim=0,
        ).model_dump(mode="json")
        error_payload["error"] = str(exc)
        await hunter.bus.publish(
            reply_channel,
            BaseEnvelope(
                kind="embedding.result.v1",
                source=_source(),
                correlation_id=env.correlation_id,
                causality_chain=env.causality_chain,
                payload=error_payload
            ),
        )
        return

    result = EmbeddingResultV1(
        doc_id=doc_id,
        embedding=embedding,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    await hunter.bus.publish(
        reply_channel,
        BaseEnvelope(
            kind="embedding.result.v1",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=result.model_dump(mode="json"),

        ),
    )

    timestamp = env.created_at.isoformat() if env.created_at else ""
    meta = _base_meta(
        env,
        original_channel=settings.VECTOR_HOST_EMBEDDING_REQUEST_CHANNEL,
        role="embedding_request",
        timestamp=timestamp,
    )
    meta.update(
        {
            "request_doc_id": doc_id,
            "requester_service": env.source.name,
        }
    )
    await _publish_semantic_upsert(
        env=env,
        text=text,
        doc_id=doc_id,
        role="embedding_request",
        meta=meta,
        embedding=embedding,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        original_channel=settings.VECTOR_HOST_EMBEDDING_REQUEST_CHANNEL,
    )


async def handle_envelope(env: BaseEnvelope) -> None:
    if env.kind == CHAT_HISTORY_MESSAGE_KIND:
        await _handle_chat_history(env)
        return
    if env.kind == "embedding.generate.v1":
        await _handle_embedding_request(env)
        return
    logger.debug("Ignoring unsupported kind=%s", env.kind)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hunter, embedder
    embedder = Embedder(settings)
    cfg = _cfg()
    hunter = Hunter(
        cfg,
        patterns=[
            settings.VECTOR_HOST_CHAT_HISTORY_CHANNEL,
            settings.VECTOR_HOST_EMBEDDING_REQUEST_CHANNEL,
        ],
        handler=handle_envelope,
    )
    await hunter.start_background()
    logger.info(
        "Vector host listening channels=%s,%s",
        settings.VECTOR_HOST_CHAT_HISTORY_CHANNEL,
        settings.VECTOR_HOST_EMBEDDING_REQUEST_CHANNEL,
    )
    yield
    if hunter:
        await hunter.stop()
    if embedder:
        await embedder.close()


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, Any]:
    bus_connected = bool(hunter and hunter.bus and getattr(hunter.bus, "_redis", None))
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "bus_connected": bus_connected,
    }
