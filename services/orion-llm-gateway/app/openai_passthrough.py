"""OpenAI-compatible HTTP passthrough for external clients (e.g. AI Town Convex actions).

Chat completions resolve Orion route keys (chat, quick, agent, …) through the same
config/gpu_pool.yaml `routes:` the bus RPC and Anthropic passthrough use; each call takes a GPU
pool lease (holder "http:openai") and goes to the granted role's URL.
Embeddings proxy to ORION_VECTOR_HOST_URL (not a GPU pool role).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from .anthropic_passthrough import (
    _extract_correlation_id,
    _httpx_timeout,
    normalize_anthropic_model_name,
)
from . import pool_placement
from .passthrough_proxy import proxy_on_pool
from .settings import settings
from .resource_lease import LeaseGuard, ResourceLeaseRejected, lease_error

logger = logging.getLogger("orion-llm-gateway.openai")


def _available_route_keys() -> list[str]:
    return sorted(pool_placement.pool_routes().keys())


def resolve_openai_route(
    requested_model: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Route key from the OpenAI `model` field. Returns (route_key, upstream_model, error_payload).

    The route must be in config/gpu_pool.yaml; the model forwarded upstream is the route key (the
    pool role's llama.cpp server serves one model and echoes its real name back)."""
    routes = pool_placement.pool_routes()
    normalized = normalize_anthropic_model_name(requested_model)
    route_key = normalized or None
    if route_key is None:
        default_key = str(settings.llm_route_default or "chat")
        if default_key in routes:
            route_key = default_key
    if route_key is None or route_key not in routes:
        label = normalized or str(requested_model or "")
        return None, None, {
            "error": {
                "message": f"OpenAI passthrough route '{label}' is not in config/gpu_pool.yaml routes",
                "type": pool_placement.ROUTE_NOT_IN_POOL,
                "available_routes": _available_route_keys(),
            }
        }
    return route_key, route_key, None


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
    route_key, upstream_model, error_payload = resolve_openai_route(
        str(requested_model) if requested_model is not None else None
    )
    if error_payload is not None:
        return JSONResponse(error_payload, status_code=404)

    assert route_key is not None and upstream_model is not None
    forward_body = dict(body)
    if forward_body.get("model") != upstream_model:
        forward_body["model"] = upstream_model
    try:
        guard = LeaseGuard.from_headers(request.headers, lane=route_key)
    except ResourceLeaseRejected as exc:
        return JSONResponse(lease_error(str(exc)), status_code=409)
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    correlation_id = _extract_correlation_id(request)

    def _log(upstream_url: str, served_by: str, stream: bool) -> None:
        logger.info("openai_chat_passthrough corr=%s route=%s upstream=%s served_by=%s stream=%s",
                    correlation_id or "-", route_key, upstream_url, served_by, stream)

    return await proxy_on_pool(
        request=request,
        route_key=route_key,
        forward_body=forward_body,
        path="/v1/chat/completions",
        holder=pool_placement.HOLDER_OPENAI,
        guard=guard,
        correlation_id=correlation_id,
        min_ctx_tokens=pool_placement.estimate_min_ctx_tokens(
            body.get("messages") or [], max_tokens,
            extra=json.dumps(body["tools"]) if body.get("tools") else None,
        ),
        anthropic=False,
        on_dispatch=_log,
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
