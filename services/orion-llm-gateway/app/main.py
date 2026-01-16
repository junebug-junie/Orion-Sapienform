from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from pydantic import ValidationError

# [FIX] Added ServiceRef to imports
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ChatResultPayload, Envelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit
from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1

from .llm_backend import run_llm_chat, run_llm_embeddings
from .models import ChatBody, EmbeddingsBody
from .settings import settings

logger = logging.getLogger("orion-llm-gateway")


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=getattr(settings, "node_name", None),
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=float(getattr(settings, "heartbeat_interval_sec", 10.0) or 10.0),
    )


# [FIX] Helper to replace the missing .service_ref() method
def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=getattr(settings, "node_name", None),
        version=settings.service_version,
    )


async def handle_chat(env: BaseEnvelope) -> BaseEnvelope:
    if env.kind not in ("llm.chat.request", "legacy.message"):
        return BaseEnvelope(
            kind="system.error",
            source=_source(),  # [FIX]
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": f"unsupported_kind:{env.kind}"},
        )

    payload_obj: Dict[str, Any] = {}
    if env.kind == "legacy.message":
        raw = env.payload if isinstance(env.payload, dict) else {}
        if raw.get("event") == "chat":
            payload_obj = raw.get("payload") or raw.get("body") or {}
        else:
            payload_obj = raw
    else:
        payload_obj = env.payload if isinstance(env.payload, dict) else {}

    try:
        typed_req = Envelope[ChatRequestPayload].model_validate(
            {**env.model_dump(), "kind": "llm.chat.request", "payload": payload_obj}
        )
    except ValidationError as ve:
        return BaseEnvelope(
            kind="llm.chat.result",
            source=_source(),  # [FIX]
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": "validation_failed", "details": ve.errors()},
        )

    body = ChatBody(
        model=typed_req.payload.model,
        messages=[m.model_dump() for m in typed_req.payload.messages],
        raw_user_text=typed_req.payload.raw_user_text,
        options=typed_req.payload.options or {},
        profile_name=typed_req.payload.profile,
        trace_id=str(typed_req.correlation_id),
        user_id=typed_req.payload.user_id,
        session_id=typed_req.payload.session_id,
        source=typed_req.source.name,
    )

    result = run_llm_chat(body)
    text = result.get("text") if isinstance(result, dict) else str(result)

    # Optional Spark/NeuralHost enrichments. These may be absent depending on
    # which gateway instance handled the request.
    spark_meta = (result.get("spark_meta") if isinstance(result, dict) else None) or {}
    spark_vector = (result.get("spark_vector") if isinstance(result, dict) else None)

    out = Envelope[ChatResultPayload](
        kind="llm.chat.result",
        source=_source(),  # [FIX]
        correlation_id=typed_req.correlation_id,
        causality_chain=typed_req.causality_chain,
        payload=ChatResultPayload(
            model_used=(result.get("raw") or {}).get("model") if isinstance(result, dict) else None,
            content=text or "",
            usage=(result.get("raw") or {}).get("usage", {}) if isinstance(result, dict) else {},
            raw=result,
            spark_meta=spark_meta,
            spark_vector=spark_vector,
        ),
    )
    return out.model_copy(update={"reply_to": None})


async def handle_embedding(env: BaseEnvelope) -> BaseEnvelope:
    if env.kind != "embedding.generate.v1":
        return BaseEnvelope(
            kind="system.error",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": f"unsupported_kind:{env.kind}"},
        )

    payload_obj: Dict[str, Any]
    if hasattr(env.payload, "model_dump"):
        payload_obj = env.payload.model_dump(mode="json")
    elif isinstance(env.payload, dict):
        payload_obj = env.payload
    else:
        payload_obj = {}

    try:
        request = EmbeddingGenerateV1.model_validate(payload_obj)
    except ValidationError as ve:
        return BaseEnvelope(
            kind="embedding.result.v1",
            source=_source(),
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={"error": "validation_failed", "details": ve.errors()},
        )

    body = EmbeddingsBody(
        input=[request.text],
        profile_name=request.embedding_profile,
        trace_id=str(env.correlation_id),
        source=env.source.name if env.source else None,
    )
    result: Dict[str, Any] = {}
    error: Optional[str] = None
    try:
        result = run_llm_embeddings(body)
    except Exception as exc:
        error = str(exc)
        logger.error("Embedding request failed for doc_id=%s: %s", request.doc_id, error)

    embedding: list[float] = []
    if isinstance(result, dict):
        data = result.get("data") or []
        if data:
            embedding = data[0].get("embedding") or []

    payload = EmbeddingResultV1(
        doc_id=request.doc_id,
        embedding=embedding,
        embedding_model=(result.get("model") if isinstance(result, dict) else None),
        embedding_dim=len(embedding) if embedding else None,
    ).model_dump(mode="json")
    if error:
        payload["error"] = error

    return BaseEnvelope(
        kind="embedding.result.v1",
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=payload,
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[LLM-GW] %(levelname)s - %(message)s")
    cfg = _cfg()
    chat_svc = Rabbit(cfg, request_channel=settings.channel_llm_intake, handler=handle_chat)
    embed_svc = Rabbit(cfg, request_channel=settings.channel_embedding_generate, handler=handle_embedding)
    logger.info(
        "Rabbit listening channels=%s,%s bus=%s",
        settings.channel_llm_intake,
        settings.channel_embedding_generate,
        cfg.bus_url,
    )
    await asyncio.gather(chat_svc.start(), embed_svc.start())


if __name__ == "__main__":
    asyncio.run(main())
