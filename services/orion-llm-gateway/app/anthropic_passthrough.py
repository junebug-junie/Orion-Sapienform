from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, Mapping, Optional, Tuple

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .llm_backend import RouteTarget, get_route_targets
from .settings import settings

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


def _normalize_backend_name(backend: Optional[str]) -> str:
    normalized = str(backend or "llamacpp").replace("_", "-").lower()
    if normalized == "llama-cpp":
        return "llamacpp"
    return normalized


def _backend_supports_anthropic_messages(backend: Optional[str]) -> bool:
    return _normalize_backend_name(backend) in _ANTHROPIC_COMPAT_BACKENDS


def _available_route_keys() -> list[str]:
    return sorted(get_route_targets().keys())


def _resolve_upstream_model(route_key: str, target: RouteTarget) -> str:
    if target.model:
        return target.model
    return route_key


def resolve_anthropic_route(
    requested_model: Optional[str],
) -> Tuple[Optional[str], Optional[RouteTarget], Optional[str], Optional[Dict[str, Any]]]:
    """
    Resolve Orion route from Anthropic model field.

    Returns (route_key, target, upstream_model, error_payload).
    """
    route_table = get_route_targets()
    if not route_table:
        return None, None, None, {
            "error": {
                "type": "route_not_configured",
                "message": "No LLM gateway route table configured",
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
                "type": "route_not_configured",
                "message": f"Anthropic passthrough route '{label}' is not configured",
                "available_routes": _available_route_keys(),
            }
        }

    if not _backend_supports_anthropic_messages(target.backend):
        backend_label = _normalize_backend_name(target.backend)
        return route_key, target, None, {
            "error": {
                "type": "backend_incompatible",
                "message": (
                    f"Route '{route_key}' backend '{backend_label}' does not expose "
                    "Anthropic Messages (/v1/messages)"
                ),
                "route": route_key,
                "backend": backend_label,
                "available_routes": _available_route_keys(),
            }
        }

    upstream_model = _resolve_upstream_model(route_key, target)
    return route_key, target, upstream_model, None


def build_models_list_payload() -> Dict[str, Any]:
    route_table = get_route_targets()
    data = [
        {
            "id": route_key,
            "type": "model",
            "display_name": route_key,
            "backend": _normalize_backend_name(target.backend),
            "served_by": target.served_by,
        }
        for route_key, target in sorted(route_table.items())
        if _backend_supports_anthropic_messages(target.backend)
    ]
    return {"data": data, "object": "list"}


def _passthrough_read_timeout_sec() -> float:
    explicit = float(getattr(settings, "llm_gateway_anthropic_passthrough_timeout_sec", 900.0) or 900.0)
    default_read = float(getattr(settings, "read_timeout_sec", 60.0) or 60.0)
    return max(30.0, min(max(explicit, default_read), 900.0))


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
    route_key, target, upstream_model, error_payload = resolve_anthropic_route(
        str(requested_model) if requested_model is not None else None
    )
    if error_payload is not None:
        err_type = error_payload.get("error", {}).get("type", "route_not_configured")
        status = 404 if err_type == "route_not_configured" else 400
        logger.warning(
            "anthropic_passthrough_error type=%s model=%s routes=%s",
            err_type,
            requested_model,
            _available_route_keys(),
        )
        return JSONResponse(error_payload, status_code=status)

    assert target is not None and route_key is not None and upstream_model is not None
    upstream_url = f"{target.url.rstrip('/')}/v1/messages"
    forward_body = dict(body)
    if forward_body.get("model") != upstream_model:
        forward_body["model"] = upstream_model

    stream = bool(forward_body.get("stream"))
    correlation_id = _extract_correlation_id(request)
    _log_passthrough_request(
        correlation_id=correlation_id,
        requested_model=str(requested_model) if requested_model is not None else None,
        route_key=route_key,
        upstream_url=upstream_url,
        served_by=target.served_by,
        stream=stream,
        body=forward_body,
    )

    headers = _forwardable_request_headers(request)
    timeout = _httpx_timeout()

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
                logger.error(
                    "anthropic_passthrough_timeout route=%s upstream=%s corr=%s",
                    route_key,
                    upstream_url,
                    correlation_id,
                )
                return JSONResponse(
                    {"error": {"type": "timeout", "message": "Upstream Anthropic Messages timed out"}},
                    status_code=504,
                )
            except httpx.HTTPError as exc:
                await client.aclose()
                logger.error(
                    "anthropic_passthrough_upstream_error route=%s upstream=%s corr=%s error=%s",
                    route_key,
                    upstream_url,
                    correlation_id,
                    exc,
                )
                return JSONResponse(
                    {"error": {"type": "upstream_error", "message": f"Upstream request failed: {exc}"}},
                    status_code=502,
                )

            if upstream.status_code >= 400:
                logger.error(
                    "anthropic_passthrough_upstream_error route=%s upstream=%s corr=%s status=%s",
                    route_key,
                    upstream_url,
                    correlation_id,
                    upstream.status_code,
                )
                response_headers = _forwardable_response_headers(upstream.headers)
                content_type = response_headers.pop("content-type", None) or response_headers.pop(
                    "Content-Type", None
                )
                error_body = await upstream.aread()
                await upstream.aclose()
                await client.aclose()
                return Response(
                    content=error_body,
                    status_code=upstream.status_code,
                    headers=response_headers,
                    media_type=content_type or "application/json",
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
            if upstream.status_code == 404:
                logger.error(
                    "anthropic_passthrough_upstream_404 route=%s upstream=%s corr=%s",
                    route_key,
                    upstream_url,
                    correlation_id,
                )
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
        logger.error(
            "anthropic_passthrough_timeout route=%s upstream=%s corr=%s",
            route_key,
            upstream_url,
            correlation_id,
        )
        return JSONResponse(
            {"error": {"type": "timeout", "message": "Upstream Anthropic Messages timed out"}},
            status_code=504,
        )
    except httpx.HTTPError as exc:
        logger.error(
            "anthropic_passthrough_upstream_error route=%s upstream=%s corr=%s error=%s",
            route_key,
            upstream_url,
            correlation_id,
            exc,
        )
        return JSONResponse(
            {"error": {"type": "upstream_error", "message": f"Upstream request failed: {exc}"}},
            status_code=502,
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
