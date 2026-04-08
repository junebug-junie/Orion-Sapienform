from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI
import uvicorn

from pydantic import ValidationError

# [FIX] Added ServiceRef to imports
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ChatResultPayload, Envelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit
from orion.schemas.vector.schemas import VectorUpsertV1

from .llm_backend import get_route_targets, run_llm_chat
from .embed_publish import publish_assistant_embedding
from .models import ChatBody
from .settings import settings

logger = logging.getLogger("orion-llm-gateway")
bus_handle: Optional[Any] = None
app = FastAPI()


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _preview_text(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    return repr(value[:limit])


@app.get("/health")
async def health() -> Dict[str, Any]:
    routes = sorted(get_route_targets().keys())
    return {
        "status": "ok",
        "service": settings.service_name,
        "node": settings.node_name,
        "routes": routes,
    }


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
        route=typed_req.payload.route,
        trace_id=str(typed_req.correlation_id),
        user_id=typed_req.payload.user_id,
        session_id=typed_req.payload.session_id,
        source=typed_req.source.name,
    )
    messages = body.messages or []
    memory_marker = "RELEVANT MEMORY"
    marked_message = next(
        (m for m in messages if memory_marker in str(getattr(m, "content", "") or "")), None
    )
    combined_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
    fallback_message = next(
        (m for m in messages if str(getattr(m, "role", "") or "").lower() == "user"), None
    )
    snippet_source = marked_message or fallback_message or (messages[0] if messages else None)
    snippet = str(getattr(snippet_source, "content", "") or "")[:160]
    logger.info(
        "llm_request_received corr_id=%s msgs_count=%s any_msg_contains_memory_marker=%s combined_chars=%s snippet=%r",
        typed_req.correlation_id,
        len(messages),
        bool(marked_message),
        combined_chars,
        snippet,
    )

    result = run_llm_chat(body)
    text = result.get("text") if isinstance(result, dict) else str(result)

    # Optional Spark/NeuralHost enrichments. These may be absent depending on
    # which gateway instance handled the request.
    spark_meta = (result.get("spark_meta") if isinstance(result, dict) else None) or {}
    spark_vector = (result.get("spark_vector") if isinstance(result, dict) else None)
    reasoning_content = (result.get("reasoning_content") if isinstance(result, dict) else None)
    inline_think_content = (result.get("inline_think_content") if isinstance(result, dict) else None)
    backend = (result.get("backend") if isinstance(result, dict) else None)
    model_used = (result.get("model") if isinstance(result, dict) else None)
    route_used = (result.get("route") if isinstance(result, dict) else None)
    served_by = (result.get("served_by") if isinstance(result, dict) else None)
    gateway_label = f"{settings.node_name or 'gateway'}-{settings.service_name}"
    meta = {
        "served_by": served_by,
        "gateway": gateway_label,
        "route": route_used,
        "provider_reasoning_available": bool(str(reasoning_content or "").strip()),
        "inline_think_extracted": bool(str(inline_think_content or "").strip()),
    }
    meta = {k: v for k, v in meta.items() if v is not None}
    if _thought_debug_enabled():
        logger.info(
            "THOUGHT_DEBUG_LLM stage=handle_chat_result corr=%s model=%s reasoning_exists=%s reasoning_len=%s content_len=%s reasoning_snippet=%r content_snippet=%r",
            typed_req.correlation_id,
            model_used,
            bool(str(reasoning_content or "").strip()),
            _debug_len(reasoning_content),
            _debug_len(text),
            _debug_snippet(reasoning_content),
            _debug_snippet(text),
        )

    reasoning_trace = (result.get("reasoning_trace") if isinstance(result, dict) else None)
    if reasoning_content and not (isinstance(reasoning_trace, dict) and str(reasoning_trace.get("content") or "").strip()):
        reasoning_trace = {
            "role": "reasoning",
            "stage": "post_answer",
            "content": reasoning_content,
        }
    out = Envelope[ChatResultPayload](
        kind="llm.chat.result",
        source=_source(),  # [FIX]
        correlation_id=typed_req.correlation_id,
        causality_chain=typed_req.causality_chain,
        payload=ChatResultPayload(
            model_used=model_used,
            content=text or "",
            reasoning_content=reasoning_content,
            reasoning_trace=reasoning_trace,
            usage=(result.get("raw") or {}).get("usage", {}) if isinstance(result, dict) else {},
            raw=(result.get("raw") if isinstance(result, dict) else None) or {},
            spark_meta=spark_meta,
            spark_vector=spark_vector,
            meta=meta or None,
        ),
    )
    trace_content = reasoning_trace.get("content") if isinstance(reasoning_trace, dict) else None
    print(
        "===THINK_HOP=== hop=llm_gateway_out "
        f"corr={typed_req.correlation_id} "
        f"has_reasoning_content={bool(reasoning_content)} "
        f"reasoning_len={len(reasoning_content) if reasoning_content else 0} "
        f"trace_len={len(trace_content) if trace_content else 0} "
        f"inline_think_len={len(inline_think_content) if isinstance(inline_think_content, str) else 0} "
        f"preview={_preview_text(reasoning_content or trace_content)}",
        flush=True,
    )
    response_payload = out.payload
    try:
        resp_keys = sorted(response_payload.model_dump().keys())
    except Exception:
        resp_keys = [type(response_payload).__name__]
    print(
        "===THINK_HOP=== hop=llm_gateway_response_shape "
        f"corr={typed_req.correlation_id} keys={resp_keys}",
        flush=True,
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


async def _serve_health() -> None:
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.llm_gateway_health_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _probe_route_targets() -> None:
    route_targets = get_route_targets()
    if not route_targets:
        logger.info("No route table configured; skipping upstream health probes.")
        return

    timeout = settings.llm_route_health_timeout_sec
    async with httpx.AsyncClient(timeout=timeout) as client:
        for route, target in route_targets.items():
            url = f"{target.url.rstrip('/')}/health"
            try:
                response = await client.get(url)
                if response.status_code >= 400:
                    logger.warning(
                        "Route '%s' health probe failed status=%s url=%s",
                        route,
                        response.status_code,
                        url,
                    )
                else:
                    logger.info("Route '%s' health probe ok url=%s", route, url)
            except Exception as exc:
                logger.warning("Route '%s' health probe error url=%s error=%s", route, url, exc)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[LLM-GW] %(levelname)s - %(message)s")
    cfg = _cfg()
    chat_svc = Rabbit(cfg, request_channel=settings.channel_llm_intake, handler=handle_chat)
    global bus_handle
    bus_handle = chat_svc.bus
    route_targets = get_route_targets()
    routes_summary = ",".join(
        f"{name}={target.url}" for name, target in sorted(route_targets.items())
    )
    logger.info(
        "[LLM-GW] startup routes=[%s] timeouts=connect:%s read:%s bus=%s channel=%s",
        routes_summary,
        settings.connect_timeout_sec,
        settings.read_timeout_sec,
        cfg.bus_url,
        settings.channel_llm_intake,
    )
    await _probe_route_targets()
    logger.info(
        "Rabbit listening channels=%s bus=%s",
        settings.channel_llm_intake,
        cfg.bus_url,
    )
    await asyncio.gather(chat_svc.start(), _serve_health())


if __name__ == "__main__":
    asyncio.run(main())
