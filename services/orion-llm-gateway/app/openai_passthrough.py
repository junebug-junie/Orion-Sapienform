"""OpenAI-compatible HTTP passthrough for external clients (e.g. AI Town Convex actions).

Chat completions resolve Orion route keys (chat, quick, agent, …) through the same
LLM_GATEWAY_ROUTE_TABLE_JSON used by cortex bus RPC and Anthropic FCC passthrough.
Embeddings proxy to ORION_LLM_OLLAMA_URL when configured (AI Town memory vectors).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .anthropic_passthrough import (
    _extract_correlation_id,
    _forwardable_request_headers,
    _forwardable_response_headers,
    _httpx_timeout,
    normalize_anthropic_model_name,
)
from .llm_backend import RouteTarget, get_route_targets
from .settings import settings

logger = logging.getLogger("orion-llm-gateway.openai")

_OPENAI_COMPAT_BACKENDS = frozenset({"llamacpp", "llama-cpp", "llama-cola", "vllm", "ollama"})


def _normalize_backend_name(backend: Optional[str]) -> str:
    normalized = str(backend or "llamacpp").replace("_", "-").lower()
    if normalized == "llama-cpp":
        return "llamacpp"
    return normalized


def _backend_supports_openai_chat(backend: Optional[str]) -> bool:
    return _normalize_backend_name(backend) in _OPENAI_COMPAT_BACKENDS


def _available_route_keys() -> list[str]:
    return sorted(get_route_targets().keys())


def _resolve_upstream_model(route_key: str, target: RouteTarget) -> str:
    if target.model:
        return target.model
    return route_key


def resolve_openai_route(
    requested_model: Optional[str],
) -> Tuple[Optional[str], Optional[RouteTarget], Optional[str], Optional[Dict[str, Any]]]:
    route_table = get_route_targets()
    if not route_table:
        return None, None, None, {
            "error": {
                "message": "No LLM gateway route table configured",
                "type": "route_not_configured",
                "available_routes": [],
            }
        }

    normalized = normalize_anthropic_model_name(requested_model)
    route_key = normalized
    target = route_table.get(route_key) if route_key else None

    if target is None and not route_key:
        default_key = str(settings.llm_route_default or "chat")
        if default_key in route_table:
            route_key = default_key
            target = route_table[default_key]

    if target is None:
        label = normalized or str(requested_model or "")
        return None, None, None, {
            "error": {
                "message": f"OpenAI passthrough route '{label}' is not configured",
                "type": "route_not_configured",
                "available_routes": _available_route_keys(),
            }
        }

    if not _backend_supports_openai_chat(target.backend):
        backend_label = _normalize_backend_name(target.backend)
        return route_key, target, None, {
            "error": {
                "message": (
                    f"Route '{route_key}' backend '{backend_label}' does not expose "
                    "OpenAI chat (/v1/chat/completions)"
                ),
                "type": "backend_incompatible",
                "route": route_key,
                "backend": backend_label,
                "available_routes": _available_route_keys(),
            }
        }

    upstream_model = _resolve_upstream_model(route_key, target)
    return route_key, target, upstream_model, None


def _disabled_response() -> JSONResponse:
    return JSONResponse(
        {"error": {"message": "OpenAI passthrough is disabled", "type": "disabled"}},
        status_code=503,
    )


def _vector_host_embedding_url() -> Optional[str]:
    base = str(getattr(settings, "orion_vector_host_url", None) or "").strip()
    if not base:
        return None
    return f"{base.rstrip('/')}/embedding"


def _embedding_texts_from_body(body: Dict[str, Any]) -> list[str]:
    raw_input = body.get("input")
    if isinstance(raw_input, list):
        return [str(v) for v in raw_input if str(v).strip()]
    return [str(raw_input or "")]


async def _proxy_upstream_json(
    *,
    request: Request,
    upstream_url: str,
    forward_body: Dict[str, Any],
    route_key: str,
    correlation_id: Optional[str],
    log_event: str,
) -> Response:
    stream = bool(forward_body.get("stream"))
    headers = _forwardable_request_headers(request)
    timeout = _httpx_timeout()

    logger.info(
        "%s corr=%s route=%s upstream=%s stream=%s",
        log_event,
        correlation_id or "-",
        route_key,
        upstream_url,
        stream,
    )

    try:
        if stream:
            client = httpx.AsyncClient(timeout=timeout)
            try:
                upstream_request = client.build_request(
                    "POST", upstream_url, headers=headers, json=forward_body
                )
                upstream = await client.send(upstream_request, stream=True)
            except httpx.TimeoutException:
                await client.aclose()
                return JSONResponse(
                    {"error": {"message": "Upstream OpenAI request timed out", "type": "timeout"}},
                    status_code=504,
                )
            except httpx.HTTPError as exc:
                await client.aclose()
                return JSONResponse(
                    {"error": {"message": f"Upstream request failed: {exc}", "type": "upstream_error"}},
                    status_code=502,
                )

            response_headers = _forwardable_response_headers(upstream.headers)
            media_type = upstream.headers.get("content-type") or "text/event-stream"

            async def _body() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()
                    await client.aclose()

            return StreamingResponse(
                _body(),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=media_type,
            )

        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream = await client.post(upstream_url, headers=headers, json=forward_body)
            response_headers = _forwardable_response_headers(upstream.headers)
            content_type = response_headers.pop("content-type", None) or response_headers.pop(
                "Content-Type", None
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=content_type or "application/json",
            )
    except httpx.TimeoutException:
        return JSONResponse(
            {"error": {"message": "Upstream OpenAI request timed out", "type": "timeout"}},
            status_code=504,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"error": {"message": f"Upstream request failed: {exc}", "type": "upstream_error"}},
            status_code=502,
        )


async def handle_chat_completions_post(request: Request) -> Response:
    if not settings.llm_gateway_openai_passthrough_enabled:
        return _disabled_response()

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            {"error": {"message": "Request body must be valid JSON", "type": "invalid_request"}},
            status_code=400,
        )
    if not isinstance(body, dict):
        return JSONResponse(
            {"error": {"message": "Request body must be a JSON object", "type": "invalid_request"}},
            status_code=400,
        )

    requested_model = body.get("model")
    route_key, target, upstream_model, error_payload = resolve_openai_route(
        str(requested_model) if requested_model is not None else None
    )
    if error_payload is not None:
        err_type = error_payload.get("error", {}).get("type", "route_not_configured")
        status = 404 if err_type == "route_not_configured" else 400
        return JSONResponse(error_payload, status_code=status)

    assert target is not None and route_key is not None and upstream_model is not None
    forward_body = dict(body)
    if forward_body.get("model") != upstream_model:
        forward_body["model"] = upstream_model
    upstream_url = f"{target.url.rstrip('/')}/v1/chat/completions"
    return await _proxy_upstream_json(
        request=request,
        upstream_url=upstream_url,
        forward_body=forward_body,
        route_key=route_key,
        correlation_id=_extract_correlation_id(request),
        log_event="openai_chat_passthrough",
    )


async def handle_embeddings_post(request: Request) -> Response:
    if not settings.llm_gateway_openai_passthrough_enabled:
        return _disabled_response()

    upstream_url = _vector_host_embedding_url()
    if not upstream_url:
        return JSONResponse(
            {
                "error": {
                    "message": "Embeddings require ORION_VECTOR_HOST_URL on the LLM gateway",
                    "type": "embeddings_not_configured",
                }
            },
            status_code=503,
        )

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            {"error": {"message": "Request body must be valid JSON", "type": "invalid_request"}},
            status_code=400,
        )
    if not isinstance(body, dict):
        return JSONResponse(
            {"error": {"message": "Request body must be a JSON object", "type": "invalid_request"}},
            status_code=400,
        )

    texts = _embedding_texts_from_body(body)
    if not texts:
        return JSONResponse(
            {"error": {"message": "Missing embedding input text", "type": "invalid_request"}},
            status_code=400,
        )

    model = str(body.get("model") or "orion-vector-host").strip()
    correlation_id = _extract_correlation_id(request)
    headers = {"Content-Type": "application/json"}
    timeout = _httpx_timeout()

    logger.info(
        "openai_embeddings_passthrough corr=%s upstream=%s model=%s count=%s",
        correlation_id or "-",
        upstream_url,
        model,
        len(texts),
    )

    data: list[Dict[str, Any]] = []
    embedding_model = model
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for index, text in enumerate(texts):
                doc_id = hashlib.sha256(f"{index}:{text}".encode("utf-8")).hexdigest()[:32]
                upstream = await client.post(
                    upstream_url,
                    headers=headers,
                    json={"doc_id": doc_id, "text": text},
                )
                if upstream.status_code >= 400:
                    return Response(
                        content=upstream.content,
                        status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type") or "application/json",
                    )
                parsed = upstream.json()
                if not isinstance(parsed, dict) or not isinstance(parsed.get("embedding"), list):
                    return JSONResponse(
                        {"error": {"message": "Invalid vector-host embedding response", "type": "upstream_error"}},
                        status_code=502,
                    )
                embedding_model = str(parsed.get("embedding_model") or model)
                data.append(
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": parsed["embedding"],
                    }
                )
    except httpx.TimeoutException:
        return JSONResponse(
            {"error": {"message": "Upstream embeddings request timed out", "type": "timeout"}},
            status_code=504,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"error": {"message": f"Upstream embeddings request failed: {exc}", "type": "upstream_error"}},
            status_code=502,
        )

    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "model": embedding_model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    )


def register_openai_passthrough_routes(app_router: Optional[APIRouter] = None) -> APIRouter:
    router = app_router or APIRouter()

    @router.post("/v1/chat/completions")
    async def post_v1_chat_completions(request: Request) -> Response:
        return await handle_chat_completions_post(request)

    @router.post("/v1/embeddings")
    async def post_v1_embeddings(request: Request) -> Response:
        return await handle_embeddings_post(request)

    return router
