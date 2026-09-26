from __future__ import annotations

import json
import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from . import pool_placement
from .passthrough_proxy import proxy_on_pool
from .settings import settings
from .resource_lease import LeaseGuard, ResourceLeaseRejected, gpu_lease_from_headers, lease_error

logger = logging.getLogger("orion-llm-gateway.anthropic")

_ANTHROPIC_COMPAT_BACKENDS = frozenset({"llamacpp", "llama-cpp"})
_PROVIDER_PREFIXES = ("llamacpp/", "orion/", "anthropic/")

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

_FORWARD_REQUEST_HEADERS = frozenset(
    {
        "content-type",
        "accept",
        "accept-encoding",
        "x-api-key",
        "anthropic-version",
        "anthropic-beta",
        "anthropic-dangerous-direct-browser-access",
        "x-request-id",
        "request-id",
    }
)


def normalize_anthropic_system_messages(body: Dict[str, Any]) -> Dict[str, Any]:
    """Hoist Claude hook system messages into Anthropic's system field.

    Claude can append SessionStart context after a user message. Native
    llama.cpp preserves that role, but its model template requires system
    context at the beginning. Preserve every block and its cache metadata,
    leaving conversational/tool order and the caller's request untouched.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(
        isinstance(message, dict) and message.get("role") == "system"
        for message in messages
    ):
        return dict(body)

    def blocks(content: Any) -> list:
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        if content is None:
            return []
        if isinstance(content, list):
            return list(content)
        raise ValueError("System content must be a string or a list of blocks")

    system = blocks(body.get("system"))
    conversation = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            additional = blocks(message.get("content"))
            if system and additional:
                system.append({"type": "text", "text": "\n\n"})
            system.extend(additional)
        else:
            conversation.append(message)
    return {**body, "system": system, "messages": conversation}


def normalize_anthropic_model_name(model: Optional[str]) -> str:
    """Strip provider prefixes; return bare Orion lane key."""
    raw = str(model or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    for prefix in _PROVIDER_PREFIXES:
        if lowered.startswith(prefix):
            return raw[len(prefix) :].strip()
    return raw


def _available_route_keys() -> list[str]:
    return sorted(pool_placement.pool_routes().keys())


def resolve_anthropic_route(
    requested_model: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Resolve Orion route from Anthropic model field.

    Returns (route_key, upstream_model, error_payload). The route must be in
    config/gpu_pool.yaml; which GPU serves it is the pool lease's grant.
    """
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
                "type": pool_placement.ROUTE_NOT_IN_POOL,
                "message": f"Anthropic passthrough route '{label}' is not in config/gpu_pool.yaml routes",
                "available_routes": _available_route_keys(),
            }
        }
    return route_key, route_key, None


def build_models_list_payload() -> Dict[str, Any]:
    """Every pool route is a llama.cpp role, so every route speaks /v1/messages. served_by is
    per call (the grant), so it is not claimed here."""
    data = [
        {
            "id": route_key,
            "type": "model",
            "display_name": route_key,
            "backend": pool_placement.LLAMACPP_BACKEND,
            "served_by": None,
        }
        for route_key in _available_route_keys()
    ]
    return {"data": data, "object": "list"}


def _passthrough_read_timeout_sec() -> float:
    explicit = float(getattr(settings, "llm_gateway_anthropic_passthrough_timeout_sec", 900.0) or 900.0)
    default_read = float(getattr(settings, "read_timeout_sec", 60.0) or 60.0)
    return max(30.0, max(explicit, default_read))


def _httpx_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=float(getattr(settings, "connect_timeout_sec", 10.0) or 10.0),
        read=_passthrough_read_timeout_sec(),
        write=10.0,
        pool=10.0,
    )


def _extract_correlation_id(request: Request) -> Optional[str]:
    for header in ("x-request-id", "request-id"):
        value = request.headers.get(header)
        if value:
            return value.strip()
    return None


def _forwardable_request_headers(request: Request) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in request.headers.items():
        lowered = key.lower()
        if lowered in _HOP_BY_HOP_HEADERS:
            continue
        if lowered in _FORWARD_REQUEST_HEADERS or lowered.startswith("anthropic-"):
            out[key] = value
    return out


def _forwardable_response_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in _HOP_BY_HOP_HEADERS:
            continue
        if lowered in {"content-type", "cache-control", "x-request-id", "request-id"} or lowered.startswith(
            "anthropic-"
        ):
            out[key] = value
    return out


def _tool_summary(body: Dict[str, Any]) -> str:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return "tools=0"
    names = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            if name:
                names.append(str(name))
    if names:
        return f"tools={len(tools)} keys={','.join(names[:5])}"
    return f"tools={len(tools)}"


def _log_passthrough_request(
    *,
    correlation_id: Optional[str],
    requested_model: Optional[str],
    route_key: str,
    upstream_url: str,
    served_by: Optional[str],
    stream: bool,
    body: Dict[str, Any],
) -> None:
    logger.info(
        "anthropic_passthrough corr=%s model=%s route=%s upstream=%s served_by=%s stream=%s %s",
        correlation_id or "-",
        requested_model or "-",
        route_key,
        upstream_url,
        served_by or "-",
        stream,
        _tool_summary(body),
    )


async def handle_messages_post(request: Request) -> Response:
    if not settings.llm_gateway_anthropic_passthrough_enabled:
        return JSONResponse(
            {"error": {"type": "disabled", "message": "Anthropic passthrough is disabled"}},
            status_code=503,
        )

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            {"error": {"type": "invalid_request", "message": "Request body must be valid JSON"}},
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": {"type": "invalid_request", "message": "Request body must be a JSON object"}},
            status_code=400,
        )

    requested_model = body.get("model")
    route_key, upstream_model, error_payload = resolve_anthropic_route(
        str(requested_model) if requested_model is not None else None
    )
    if error_payload is not None:
        logger.warning(
            "anthropic_passthrough_error type=%s model=%s routes=%s",
            error_payload.get("error", {}).get("type"),
            requested_model,
            _available_route_keys(),
        )
        return JSONResponse(error_payload, status_code=404)

    assert route_key is not None and upstream_model is not None
    try:
        forward_body = normalize_anthropic_system_messages(body)
    except ValueError as exc:
        return JSONResponse(
            {"error": {"type": "invalid_request", "message": str(exc)}}, status_code=400
        )
    try:
        guard = LeaseGuard.from_headers(request.headers, lane=route_key)
        hold = gpu_lease_from_headers(request.headers)
    except ResourceLeaseRejected as exc:
        return JSONResponse(lease_error(str(exc)), status_code=409)
    if forward_body.get("model") != upstream_model:
        forward_body["model"] = upstream_model

    correlation_id = _extract_correlation_id(request)

    def _log(upstream_url: str, served_by: str, stream: bool) -> None:
        _log_passthrough_request(
            correlation_id=correlation_id,
            requested_model=str(requested_model) if requested_model is not None else None,
            route_key=route_key,
            upstream_url=upstream_url,
            served_by=served_by,
            stream=stream,
            body=forward_body,
        )

    return await proxy_on_pool(
        request=request,
        route_key=route_key,
        forward_body=forward_body,
        path="/v1/messages",
        holder=pool_placement.HOLDER_ANTHROPIC,
        guard=guard,
        hold=hold,
        correlation_id=correlation_id,
        min_ctx_tokens=pool_placement.estimate_min_ctx_tokens(
            forward_body.get("messages") or [], forward_body.get("max_tokens"),
            extra=[forward_body.get("system"), json.dumps(forward_body["tools"]) if forward_body.get("tools") else None],
        ),
        anthropic=True,
        on_dispatch=_log,
    )


def handle_messages_get() -> Response:
    return handle_messages_head()


def handle_messages_head() -> Response:
    if not settings.llm_gateway_anthropic_passthrough_enabled:
        return Response(status_code=503)
    return Response(status_code=200)


def handle_messages_options() -> Response:
    if not settings.llm_gateway_anthropic_passthrough_enabled:
        return Response(status_code=503)
    return Response(
        status_code=204,
        headers={
            "Allow": "GET, POST, HEAD, OPTIONS",
            "Access-Control-Allow-Methods": "POST, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "content-type, x-api-key, anthropic-version, anthropic-beta",
        },
    )


def handle_models_get() -> JSONResponse:
    if not settings.llm_gateway_anthropic_passthrough_enabled:
        return JSONResponse(
            {"error": {"type": "disabled", "message": "Anthropic passthrough is disabled"}},
            status_code=503,
        )
    return JSONResponse(build_models_list_payload())


def register_anthropic_passthrough_routes(app_router: Optional[APIRouter] = None) -> APIRouter:
    router = app_router or APIRouter()

    @router.get("/v1/models")
    async def get_v1_models() -> JSONResponse:
        return handle_models_get()

    @router.get("/v1/messages")
    async def get_v1_messages() -> Response:
        return handle_messages_get()

    @router.post("/v1/messages")
    async def post_v1_messages(request: Request) -> Response:
        return await handle_messages_post(request)

    @router.head("/v1/messages")
    async def head_v1_messages() -> Response:
        return handle_messages_head()

    @router.options("/v1/messages")
    async def options_v1_messages() -> Response:
        return handle_messages_options()

    return router
