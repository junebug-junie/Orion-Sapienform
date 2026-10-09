"""Stop a blocking upstream call from outside the thread that is making it.

The bus path runs ``run_llm_chat`` (sync httpx) in a per-role thread. When the GPU pool takes the
lease back (lost, or recalled and the grace ran out) or the caller's budget runs out, the slot the
pool now thinks is free must actually be free -- so the in-flight request has to stop.

``httpx.Client.close()`` from another thread does NOT do that: it closes the pool's bookkeeping but
a thread blocked in ``recv()`` stays blocked (checked: a request to a server that never answers
stays stuck after ``close()``). ``socket.shutdown(SHUT_RDWR)`` does wake it. So the worker's HTTP
client is built on a network backend that records every socket it opens into a per-call
``UpstreamCancel``; ``cancel()`` shuts those sockets down from the event loop, the blocked read
fails, and the worker returns.

Wiring: ``run_cancellable(handle, fn, ...)`` sets the handle for the worker thread;
``llm_backend._common_http_client`` asks ``cancellable_transport()`` for a transport. Outside a
cancellable call (unit tests, direct calls) nothing changes.
"""
from __future__ import annotations

import socket
import threading
import time
from typing import Any, Callable, List, Optional, TypeVar

import httpcore
import httpx

T = TypeVar("T")

_local = threading.local()


class UpstreamCancel:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sockets: List[socket.socket] = []
        self.cancelled = threading.Event()
        self.reason: Optional[str] = None
        # Monotonic clock: when the call began, and when the gateway hung up on it. llm_backend
        # logs both so a cancelled call's line says how long it ran and how soon after the cancel
        # the worker thread came out.
        self.started_at = time.monotonic()
        self.cancelled_at: Optional[float] = None

    def register(self, sock: Optional[socket.socket]) -> None:
        if sock is None:
            return
        with self._lock:
            if not self.cancelled.is_set():
                self._sockets.append(sock)
                return
        _shutdown(sock)  # opened after cancel: stop it at once

    def cancel(self, reason: str) -> None:
        with self._lock:
            if self.cancelled.is_set():
                return
            self.reason = reason
            self.cancelled_at = time.monotonic()
            self.cancelled.set()
            sockets, self._sockets = self._sockets, []
        for sock in sockets:
            _shutdown(sock)


def _shutdown(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # already closed


class _RecordingBackend(httpcore.SyncBackend):
    def __init__(self, handle: UpstreamCancel) -> None:
        super().__init__()
        self._handle = handle

    def connect_tcp(self, *args: Any, **kwargs: Any) -> httpcore.NetworkStream:
        if self._handle.cancelled.is_set():
            raise httpcore.ConnectError(f"upstream call cancelled: {self._handle.reason}")
        stream = super().connect_tcp(*args, **kwargs)
        self._handle.register(stream.get_extra_info("socket"))
        return stream


def current() -> Optional[UpstreamCancel]:
    return getattr(_local, "handle", None)


def cancelled_by_gateway() -> Optional[UpstreamCancel]:
    """The current call's handle if the gateway itself has hung up on it, else None.

    llm_backend asks this before blaming the worker for a failed request: once the gateway has shut
    the sockets down, the RemoteProtocolError / ReadError / ConnectError that follows is the
    gateway's own doing, not the upstream's."""
    handle = current()
    if handle is not None and handle.cancelled.is_set():
        return handle
    return None


def run_cancellable(handle: UpstreamCancel, fn: Callable[..., T], *args: Any) -> T:
    """Run ``fn`` in this (worker) thread with ``handle`` as its cancel hook."""
    _local.handle = handle
    try:
        return fn(*args)
    finally:
        _local.handle = None


def cancellable_transport() -> Optional[httpx.HTTPTransport]:
    """A transport whose sockets the current call's handle can shut down, or None outside one.

    Swaps the connection pool's network backend: httpx 0.27 does not expose ``network_backend``
    on HTTPTransport. tests/test_pool_lease_lifecycle.py drives a real socket through this, so an
    httpx/httpcore upgrade that breaks it fails there rather than silently not cancelling."""
    handle = current()
    if handle is None:
        return None
    transport = httpx.HTTPTransport()
    transport._pool._network_backend = _RecordingBackend(handle)  # noqa: SLF001
    return transport
