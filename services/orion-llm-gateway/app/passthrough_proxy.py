"""Shared lease-wrapped proxy for the OpenAI and Anthropic HTTP passthroughs.

Each request takes a GPU pool lease for its route's class, is POSTed to ``<grant.url><path>``,
and releases the lease when the response is done -- for a stream, when the stream ends, errors,
or the client goes away (LeaseStreamingResponse + stream_cleanup). A context overflow re-leases
once with a larger ``min_ctx_tokens``, same as the bus path. The pool wait is
LLM_GATEWAY_POOL_PASSTHROUGH_WAIT_SEC and a queued acquire is withdrawn if the client disconnects.

A request carrying ``X-Orion-Gpu-Lease`` (stage 4: FCC under a durable run's hold) attaches to that
hold instead of acquiring; its overflow is returned as is, since the child cannot leave the hold's role.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, Optional

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from orion.gpu_pool.client import Lease, LeaseUnavailable
from orion.schemas.gpu_pool import GpuLeaseRefV1

from .ctx_overflow import is_context_overflow
from .pool_placement import (
    POOL_RECALLED,
    POOL_UNAVAILABLE,
    LeaseStreamingResponse,
    PoolLease,
    class_max_ctx,
    mark_revoked,
    passthrough_wait_sec,
    route_spec,
    stream_cleanup,
    wait_lease_revoked,
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


def _mark_overflow(lease: Lease) -> None:
    lease.release_outcome, lease.release_detail = "ok", "context_overflow"


def _plain_response(content: bytes, upstream: httpx.Response) -> Response:
    headers = _http_helpers()[1](upstream.headers)
    content_type = headers.pop("content-type", None) or headers.pop("Content-Type", None)
    return Response(content=content, status_code=upstream.status_code, headers=headers,
                    media_type=content_type or "application/json")


async def _guarded(guard: Optional[LeaseGuard], operation):
    return await (guard.run(operation) if guard is not None else operation)


# How often a queued passthrough checks whether its HTTP client is still there.
_DISCONNECT_POLL_SEC = 0.5
CLIENT_CLOSED_STATUS = 499  # nginx's "client closed request": nobody reads it, the logs do


class _Revoked(Exception):
    """The pool took the lease back (lost, or recalled past its grace) mid-call."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _ClientGone(Exception):
    """The HTTP client disconnected while its acquire was still queued."""


async def _race(awaitable, watch: "asyncio.Future[str]"):
    """Await ``awaitable`` unless ``watch`` (pool_placement.wait_lease_revoked) fires first; then
    cancel it -- which closes the in-flight upstream request -- and raise _Revoked."""
    task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if task.done():
        return task.result()
    task.cancel()
    with contextlib.suppress(BaseException):
        await task
    raise _Revoked(watch.result())


async def _acquire(handle: PoolLease, request: Request) -> Lease:
    """Acquire, but withdraw the queued lease if the HTTP client goes away while waiting."""
    task = asyncio.ensure_future(handle.acquire())
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=_DISCONNECT_POLL_SEC)
            if done:
                return task.result()
            if await request.is_disconnected():
                task.cancel()  # gpu_lease withdraws the queued request on cancellation
                with contextlib.suppress(BaseException):
                    await task
                raise _ClientGone()
    except asyncio.CancelledError:
        task.cancel()
        raise


def _recalled_body(reason: str, route_key: str) -> Dict[str, Any]:
    return error_body(POOL_RECALLED, f"GPU pool took the lease for route '{route_key}' back ({reason}); "
                      "the upstream call was stopped", reason=f"lease_{reason}", route=route_key)


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
    hold: Optional[GpuLeaseRefV1] = None,
) -> Response:
    """Same placement rules as the bus path (main._dispatch_on_pool): at most three acquires --
    one clamp to the class's largest ctx on ``min_ctx_exceeds_class``, one re-lease after a real
    overflow, and the overflow response itself when nothing bigger exists.

    Lease revocation: a lost lease, or a recall whose grace has run out, stops the upstream call
    (the in-flight request is cancelled / the stream closed), the client gets a ``gpu_pool_recalled``
    error (a 503, or an SSE error event mid-stream) and the lease is released ``cancelled``."""
    spec = route_spec(route_key)
    stream = bool(forward_body.get("stream"))
    forwardable_request_headers, forwardable_response_headers, httpx_timeout = _http_helpers()
    headers = forwardable_request_headers(request)
    timeout = httpx_timeout()
    min_ctx = int(min_ctx_tokens)
    overflow_response: Optional[Response] = None
    clamped = False
    if guard is not None:
        # A stale durable token never takes a GPU lease.
        try:
            await guard.check()
        except ResourceLeaseRejected as exc:
            return JSONResponse(lease_error(str(exc)), status_code=409)

    for _ in range(3):
        handle = PoolLease(route=route_key, spec=spec, holder=holder, turn_correlation_id=correlation_id,
                           min_ctx_tokens=min_ctx, deadline_sec=passthrough_wait_sec(), hold=hold)
        try:
            lease = await _acquire(handle, request)
        except _ClientGone:
            logger.info("passthrough_client_gone_while_queued route=%s holder=%s corr=%s",
                        route_key, holder, correlation_id or "-")
            return Response(status_code=CLIENT_CLOSED_STATUS)
        except LeaseUnavailable as exc:
            if overflow_response is not None:
                return overflow_response
            max_ctx = class_max_ctx(exc.reason)
            if max_ctx is not None and not clamped and max_ctx < min_ctx:
                logger.warning("passthrough_min_ctx_exceeds_class route=%s estimate=%s -> clamp min_ctx=%s corr=%s",
                               route_key, min_ctx, max_ctx, correlation_id or "-")
                clamped, min_ctx = True, max_ctx
                continue
            logger.warning("passthrough_gpu_pool_unavailable route=%s class=%s holder=%s reason=%s corr=%s",
                           route_key, spec.work_class, holder, exc.reason, correlation_id or "-")
            return JSONResponse(error_body(
                POOL_UNAVAILABLE, f"GPU pool could not place route '{route_key}': {exc.reason}",
                reason=exc.reason, route=route_key, work_class=spec.work_class,
            ), status_code=503)

        may_release = overflow_response is None and not clamped and hold is None
        upstream_url = f"{lease.grant.url.rstrip('/')}{path}"
        if on_dispatch is not None:
            on_dispatch(upstream_url, lease.grant.served_by, stream)
        watch = asyncio.ensure_future(wait_lease_revoked(lease))
        stream_owns_lease = False
        try:
            if stream:
                client = httpx.AsyncClient(timeout=timeout)
                try:
                    upstream_request = client.build_request("POST", upstream_url, headers=headers, json=forward_body)
                    upstream = await _race(_guarded(guard, client.send(upstream_request, stream=True)), watch)
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
                    if is_context_overflow(upstream.status_code, _json_or_none(content)):
                        # Too big for the slot is not the GPU's failure: keep it out of the error counts.
                        _mark_overflow(lease)
                    if may_release and lease.release_detail == "context_overflow":
                        overflow_response = response
                        min_ctx = int(lease.grant.ctx_per_slot or min_ctx) + 1
                        await handle.release(failure)
                        continue
                    await handle.release(failure)
                    return response

                close_stream = stream_cleanup(upstream, client, handle)
                stream_watch = watch

                async def close(error: Optional[BaseException] = None) -> None:
                    stream_watch.cancel()
                    await close_stream(error)

                async def _body() -> AsyncIterator[bytes]:
                    try:
                        chunks = upstream.aiter_bytes()
                        if guard is not None:
                            chunks = guard.chunks(chunks)
                        iterator = chunks.__aiter__()
                        while True:
                            try:
                                chunk = await _race(iterator.__anext__(), stream_watch)
                            except StopAsyncIteration:
                                break
                            yield chunk
                    except _Revoked as exc:
                        mark_revoked(lease, exc.reason)
                        logger.warning("passthrough_lease_revoked_mid_stream route=%s lease_id=%s reason=%s corr=%s",
                                       route_key, lease.lease_id, exc.reason, correlation_id or "-")
                        yield _sse_error(_recalled_body(exc.reason, route_key), anthropic=anthropic)
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
                upstream = await _race(_guarded(guard, client.post(upstream_url, headers=headers, json=forward_body)),
                                       watch)
            response = _plain_response(upstream.content, upstream)
            if upstream.status_code >= 400:
                failure = _UpstreamStatus(f"http_{upstream.status_code}")
                if is_context_overflow(upstream.status_code, _json_or_none(upstream.content)):
                    # Too big for the slot is not the GPU's failure: keep it out of the error counts.
                    _mark_overflow(lease)
                if may_release and lease.release_detail == "context_overflow":
                    overflow_response = response
                    min_ctx = int(lease.grant.ctx_per_slot or min_ctx) + 1
                    await handle.release(failure)
                    continue
                await handle.release(failure)
                return response
            await handle.release()
            return response
        except _Revoked as exc:
            mark_revoked(lease, exc.reason)
            await handle.release()
            logger.warning("passthrough_lease_revoked route=%s lease_id=%s reason=%s corr=%s",
                           route_key, lease.lease_id, exc.reason, correlation_id or "-")
            return JSONResponse(_recalled_body(exc.reason, route_key), status_code=503)
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
        finally:
            if not stream_owns_lease:
                watch.cancel()
    return overflow_response or JSONResponse(error_body(POOL_UNAVAILABLE, "overflow retry exhausted"), status_code=503)
