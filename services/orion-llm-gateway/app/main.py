from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from pydantic import ValidationError

# [FIX] Added ServiceRef to imports
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ChatResultPayload, Envelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit
from orion.schemas.vector.schemas import VectorUpsertV1

from .llm_backend import run_llm_chat
from .embed_publish import publish_assistant_embedding
from .models import ChatBody
from .settings import settings

logger = logging.getLogger("orion-llm-gateway")
bus_handle: Optional[Any] = None


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


async def _maybe_publish_latent_upsert(
    *,
    env: BaseEnvelope,
    spark_vector: Optional[list[float]],
    backend: Optional[str],
    model_used: Optional[str],
    session_id: Optional[str],
    user_id: Optional[str],
) -> None:
    if not spark_vector:
        return
    if backend not in ("vllm", "llama-cola"):
        return
    if not bus_handle or not getattr(bus_handle, "enabled", False):
        logger.warning("Latent upsert skipped: bus unavailable.")
        return

    doc_id = str(env.correlation_id or env.id)
    meta: Dict[str, Any] = {
        "source_service": env.source.name,
        "original_channel": settings.channel_llm_intake,
        "role": "assistant",
        "timestamp": env.created_at.isoformat() if env.created_at else None,
        "correlation_id": str(env.correlation_id),
        "backend": backend,
        "model_used": model_used,
        "session_id": session_id,
        "user_id": user_id,
        "envelope_id": str(env.id),
    }
    meta = {k: v for k, v in meta.items() if v is not None}

    upsert = VectorUpsertV1(
        doc_id=doc_id,
        collection=settings.orion_vector_latent_collection,
        embedding=spark_vector,
        embedding_kind="latent",
        embedding_model=model_used,
        embedding_dim=len(spark_vector),
        text=None,
        meta=meta,
    )
    envelope = BaseEnvelope(
        kind="vector.upsert.v1",
        source=_source(),
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload=upsert.model_dump(mode="json"),
    )
    try:
        await bus_handle.publish(settings.channel_vector_latent_upsert, envelope)
    except Exception as exc:
        logger.warning("Latent upsert publish failed doc_id=%s error=%s", doc_id, exc)


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
    backend = (result.get("backend") if isinstance(result, dict) else None)
    model_used = (result.get("model") if isinstance(result, dict) else None)

    out = Envelope[ChatResultPayload](
        kind="llm.chat.result",
        source=_source(),  # [FIX]
        correlation_id=typed_req.correlation_id,
        causality_chain=typed_req.causality_chain,
        payload=ChatResultPayload(
            model_used=model_used,
            content=text or "",
            usage=(result.get("raw") or {}).get("usage", {}) if isinstance(result, dict) else {},
            raw=(result.get("raw") if isinstance(result, dict) else None) or {},
            spark_meta=spark_meta,
            spark_vector=spark_vector,
        ),
    )
    if bus_handle and text:
        doc_id = str(typed_req.correlation_id or env.id)
        try:
            asyncio.create_task(
                publish_assistant_embedding(
                    bus_handle,
                    text=text,
                    doc_id=doc_id,
                    trace_id=typed_req.correlation_id,
                )
            )
        except Exception as exc:
            logger.warning("Embedding publish schedule failed doc_id=%s error=%s", doc_id, exc)
    await _maybe_publish_latent_upsert(
        env=env,
        spark_vector=spark_vector,
        backend=backend,
        model_used=model_used or out.payload.model_used,
        session_id=typed_req.payload.session_id,
        user_id=typed_req.payload.user_id,
    )
    return out.model_copy(update={"reply_to": None})


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[LLM-GW] %(levelname)s - %(message)s")
    cfg = _cfg()
    chat_svc = Rabbit(cfg, request_channel=settings.channel_llm_intake, handler=handle_chat)
    global bus_handle
    bus_handle = chat_svc.bus
    logger.info(
        "Rabbit listening channels=%s bus=%s",
        settings.channel_llm_intake,
        cfg.bus_url,
    )
    await asyncio.gather(chat_svc.start())


if __name__ == "__main__":
    asyncio.run(main())
