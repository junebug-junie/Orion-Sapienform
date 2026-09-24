"""Per-hop RPC-health timing for outbound HTTP (httpx).

A0 / "Mesh transport coverage" of
docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md:
HTTP hops (thought -> mind, durable-runs -> gateway, fcc proxy -> gateway, ...) record
into the same per-bus RPC-health aggregator ``rpc_request()`` uses, under hop key
``http:<host>[:<port>]<path>``, so ``rpc_health_publish_loop`` reports them in
``RpcHealthSnapshotV1.channel_latency``.

Usage (async)::

    from orion.core.bus.http_health import AsyncHopTimingTransport

    client = httpx.AsyncClient(
        transport=AsyncHopTimingTransport(httpx.AsyncHTTPTransport(), recorder_getter=lambda: rpc_bus),
        timeout=30.0,
    )

Sync clients use ``HopTimingTransport`` the same way. ``recorder_getter`` is called per
request (same reason ``rpc_health_publish_loop`` takes a ``bus_getter``: the bus the
publish loop drains may be created after the client). It returns anything with
``record_hop_success(hop, elapsed_ms)`` / ``record_hop_timeout(hop, elapsed_ms)`` --
an ``OrionBusAsync`` or an ``RpcHealthAggregator`` -- or ``None`` to skip recording.

Outcome mapping (deliberately transport-level, not application-level):

- any HTTP response except 504 -> success with its wall time (a 4xx/5xx still made the
  round trip; application errors are not transport health)
- ``httpx.TimeoutException`` or a 504 response -> timeout
- any other exception (connection refused, DNS, ...) -> not recorded as latency; it is
  re-raised untouched. A refused connection is liveness, not latency, and is already
  visible to the caller.

Elapsed time is measured to response HEADERS (when the transport returns), not to the
end of a streamed body. For ordinary non-streaming calls against services that answer
after doing the work, that is the round trip; for a streamed response it is time to
first byte.

Thread safety: the aggregator is lock-free and assumes the asyncio event-loop thread.
Use the sync ``HopTimingTransport`` only from that thread, not from ``run_in_executor``
workers.

A transport wrapper, not httpx ``event_hooks``: response hooks never fire on a timeout,
which is exactly the outcome transport health most needs. Recording never raises into
the request path.

Passing ``transport=`` to an httpx client disables httpx's env-proxy mounts
(``HTTP(S)_PROXY`` with ``trust_env``); the wrapped hops in this repo are tailnet calls with
no proxy configured (checked for orion-thought and orion-durable-runs 2026-09-24). Test code
that injects its own ``transport=`` into a client built with these kwargs will collide --
wire the recorder only where no test double already owns the transport.

``path_normalizer`` collapses high-cardinality paths (ids in the URL) to a stable key,
e.g. ``lambda p: re.sub(r"/[0-9a-f-]{16,}", "/:id", p)``. Key cardinality is also capped
by the aggregator (``MAX_DISTINCT_HOPS``, overflow folded into ``_overflow``).
"""
from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Callable, Optional, Protocol

import httpx

logger = logging.getLogger("orion.bus.http_health")


class HopRecorder(Protocol):
    def record_hop_success(self, hop: str, elapsed_ms: float) -> None: ...

    def record_hop_timeout(self, hop: str, elapsed_ms: Optional[float] = None) -> None: ...


RecorderGetter = Callable[[], Optional[HopRecorder]]
PathNormalizer = Callable[[str], str]


_ID_SEGMENT_RE = re.compile(
    r"^(?:\d+|[0-9a-fA-F]{16,}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)


def normalize_id_path(path: str) -> str:
    """Default ``path_normalizer``: collapse path segments that look like ids -- all
    digits, a UUID, or 16+ hex chars -- to ``:id``, so ``/runs/<uuid>/cancel`` and
    ``/runs/<other-uuid>/cancel`` share one hop key. Named segments (a lane or model name
    like ``qwen3.5-27b``) are kept: they are bounded and the split is the point."""
    return "/".join(":id" if seg and _ID_SEGMENT_RE.match(seg) else seg for seg in path.split("/"))


def http_hop_key(url: httpx.URL | str, path_normalizer: Optional[PathNormalizer] = None) -> str:
    """``http:<host>[:<port>]<path>`` -- port only when explicit in the URL; no query string."""
    u = url if isinstance(url, httpx.URL) else httpx.URL(str(url))
    host = u.host or ""
    if u.port is not None:
        host = f"{host}:{u.port}"
    path = u.path or "/"
    if path_normalizer is not None:
        try:
            path = path_normalizer(path)
        except Exception:
            logger.warning("http_health path_normalizer failed path=%s", path, exc_info=True)
    return f"http:{host}{path}"


def _record(
    recorder_getter: RecorderGetter,
    hop: str,
    elapsed_ms: float,
    *,
    timed_out: bool,
) -> None:
    try:
        recorder = recorder_getter()
        if recorder is None:
            return
        if timed_out:
            recorder.record_hop_timeout(hop, elapsed_ms)
        else:
            recorder.record_hop_success(hop, elapsed_ms)
    except Exception:
        logger.warning("http_health record failed hop=%s", hop, exc_info=True)


class AsyncHopTimingTransport(httpx.AsyncBaseTransport):
    """Wraps an ``httpx.AsyncBaseTransport``; records each request's outcome as a hop."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        *,
        recorder_getter: RecorderGetter,
        path_normalizer: Optional[PathNormalizer] = None,
    ) -> None:
        self._inner = inner
        self._recorder_getter = recorder_getter
        self._path_normalizer = path_normalizer

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        hop = http_hop_key(request.url, self._path_normalizer)
        started = perf_counter()
        try:
            response = await self._inner.handle_async_request(request)
        except httpx.TimeoutException:
            _record(self._recorder_getter, hop, (perf_counter() - started) * 1000.0, timed_out=True)
            raise
        _record(
            self._recorder_getter,
            hop,
            (perf_counter() - started) * 1000.0,
            timed_out=response.status_code == 504,
        )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


class HopTimingTransport(httpx.BaseTransport):
    """Sync counterpart of ``AsyncHopTimingTransport``."""

    def __init__(
        self,
        inner: httpx.BaseTransport,
        *,
        recorder_getter: RecorderGetter,
        path_normalizer: Optional[PathNormalizer] = None,
    ) -> None:
        self._inner = inner
        self._recorder_getter = recorder_getter
        self._path_normalizer = path_normalizer

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        hop = http_hop_key(request.url, self._path_normalizer)
        started = perf_counter()
        try:
            response = self._inner.handle_request(request)
        except httpx.TimeoutException:
            _record(self._recorder_getter, hop, (perf_counter() - started) * 1000.0, timed_out=True)
            raise
        _record(
            self._recorder_getter,
            hop,
            (perf_counter() - started) * 1000.0,
            timed_out=response.status_code == 504,
        )
        return response

    def close(self) -> None:
        self._inner.close()
