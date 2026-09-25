"""Shared lease-wrapped proxy for the OpenAI and Anthropic HTTP passthroughs.

Each request takes a GPU pool lease for its route's class, is POSTed to ``<grant.url><path>``,
and releases the lease when the response is done -- for a stream, when the stream ends, errors,
or the client goes away (LeaseStreamingResponse + stream_cleanup). A context overflow re-leases
once with a larger ``min_ctx_tokens``, same as the bus path.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, Optional

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from orion.gpu_pool.client import LeaseUnavailable

from .ctx_overflow import is_context_overflow
from .pool_placement import (
    POOL_UNAVAILABLE,
    LeaseStreamingResponse,
    PoolLease,
    route_spec,
    stream_cleanup,
    wait_budget_sec,
)
from .resource_lease import LeaseGuard, ResourceLeaseRejected, lease_error

logger = logging.getLogger("orion-llm-gateway.passthrough")


class _UpstreamStatus(Exception):
    """Carried into the lease release so the pool records ``upstream_error``."""


def error_body(err_type: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"error": {"type": err_type, "message": message, **extra}}


def _sse_error(payload: Dict[str, Any], *, anthropic: bool) -> bytes:
    if anthropic:
        return ("event: error\ndata: " + json.dumps({"type": "error", **payload}) + "\n\n").encode()
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def _json_or_none(content: bytes) -> Any:
    try:
        return json.loads(content or b"null")
    except (ValueError, UnicodeDecodeError):
        return None


def _http_helpers():
    # Lazy: anthropic_passthrough imports this module.
    from .anthropic_passthrough import _forwardable_request_headers, _forwardable_response_headers, _httpx_timeout

    return _forwardable_request_headers, _forwardable_response_headers, _httpx_timeout


def _plain_response(content: bytes, upstream: httpx.Response) -> Response:
    headers = _http_helpers()[1](upstream.headers)
    content_type = headers.pop("content-type", None) or headers.pop("Content-Type", None)
    return Response(content=content, status_code=upstream.status_code, headers=headers,
                    media_type=content_type or "application/json")


async def _guarded(guard: Optional[LeaseGuard], operation):
    return await (guard.run(operation) if guard is not None else operation)


async def proxy_on_pool(
    *,
    request: Request,
    route_key: str,
    forward_body: Dict[str, Any],
    path: str,
    holder: str,
    guard: Optional[LeaseGuard],
    correlation_id: Optional[str],
    min_ctx_tokens: int,
    anthropic: bool,
    on_dispatch: Optional[Callable[[str, str, bool], None]] = None,
) -> Response:
    spec = route_spec(route_key)
    stream = bool(forward_body.get("stream"))
    forwardable_request_headers, forwardable_response_headers, httpx_timeout = _http_helpers()
    headers = forwardable_request_headers(request)
    timeout = httpx_timeout()
    min_ctx = int(min_ctx_tokens)
    overflow_response: Optional[Response] = None
    if guard is not None:
        # A stale durable token never takes a GPU lease.
        try:
            await guard.check()
        except ResourceLeaseRejected as exc:
            return JSONResponse(lease_error(str(exc)), status_code=409)

    for attempt in (1, 2):
        handle = PoolLease(route=route_key, spec=spec, holder=holder, turn_correlation_id=correlation_id,
                           min_ctx_tokens=min_ctx, deadline_sec=wait_budget_sec(spec.priority))
        try:
            lease = await handle.acquire()
        except LeaseUnavailable as exc:
            if overflow_response is not None:
                return overflow_response
            logger.warning("passthrough_gpu_pool_unavailable route=%s class=%s holder=%s reason=%s corr=%s",
                           route_key, spec.work_class, holder, exc.reason, correlation_id or "-")
            return JSONResponse(error_body(
                POOL_UNAVAILABLE, f"GPU pool could not place route '{route_key}': {exc.reason}",
                reason=exc.reason, route=route_key, work_class=spec.work_class,
            ), status_code=503)

        upstream_url = f"{lease.grant.url.rstrip('/')}{path}"
        if on_dispatch is not None:
            on_dispatch(upstream_url, lease.grant.served_by, stream)
        stream_owns_lease = False
        try:
            if stream:
                client = httpx.AsyncClient(timeout=timeout)
                try:
                    upstream_request = client.build_request("POST", upstream_url, headers=headers, json=forward_body)
                    upstream = await _guarded(guard, client.send(upstream_request, stream=True))
                except BaseException:
                    await client.aclose()
                    raise
                if upstream.status_code >= 400:
                    try:
                        content = await _guarded(guard, upstream.aread())
                    finally:
                        await upstream.aclose()
                        await client.aclose()
                    response = _plain_response(content, upstream)
                    failure = _UpstreamStatus(f"http_{upstream.status_code}")
                    if attempt == 1 and is_context_overflow(upstream.status_code, _json_or_none(content)):
                        overflow_response = response
                        min_ctx = int(lease.grant.ctx_per_slot or min_ctx) + 1
                        await handle.release(failure)
                        continue
                    await handle.release(failure)
                    return response

                close = stream_cleanup(upstream, client, handle)

                async def _body() -> AsyncIterator[bytes]:
                    try:
                        chunks = upstream.aiter_bytes()
                        if guard is not None:
                            chunks = guard.chunks(chunks)
                        async for chunk in chunks:
                            yield chunk
                    except ResourceLeaseRejected as exc:
                        yield _sse_error(lease_error(str(exc)), anthropic=anthropic)
                    except httpx.HTTPError as exc:
                        await close(exc)
                        raise
                    finally:
                        await close()

                stream_owns_lease = True
                return LeaseStreamingResponse(
                    _body(),
                    cleanup=close,
                    status_code=upstream.status_code,
                    headers=forwardable_response_headers(upstream.headers),
                    media_type=upstream.headers.get("content-type") or "text/event-stream",
                )

            async with httpx.AsyncClient(timeout=timeout) as client:
                upstream = await _guarded(guard, client.post(upstream_url, headers=headers, json=forward_body))
            response = _plain_response(upstream.content, upstream)
            if upstream.status_code >= 400:
                failure = _UpstreamStatus(f"http_{upstream.status_code}")
                if attempt == 1 and is_context_overflow(upstream.status_code, _json_or_none(upstream.content)):
                    overflow_response = response
                    min_ctx = int(lease.grant.ctx_per_slot or min_ctx) + 1
                    await handle.release(failure)
                    continue
                await handle.release(failure)
                return response
            await handle.release()
            return response
        except ResourceLeaseRejected as exc:
            await handle.release()
            return JSONResponse(lease_error(str(exc)), status_code=409)
        except httpx.TimeoutException as exc:
            await handle.release(exc)
            logger.error("passthrough_timeout route=%s upstream=%s corr=%s", route_key, upstream_url, correlation_id)
            return JSONResponse(error_body("timeout", "Upstream request timed out"), status_code=504)
        except httpx.HTTPError as exc:
            await handle.release(exc)
            logger.error("passthrough_upstream_error route=%s upstream=%s corr=%s error=%s",
                         route_key, upstream_url, correlation_id, exc)
            return JSONResponse(error_body("upstream_error", f"Upstream request failed: {exc}"), status_code=502)
        except BaseException as exc:
            if not stream_owns_lease:
                await handle.release(exc)
            raise
    return overflow_response or JSONResponse(error_body(POOL_UNAVAILABLE, "overflow retry exhausted"), status_code=503)
