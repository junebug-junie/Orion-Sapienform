"""Governor-local Anthropic relay for the warm FCC chat pool.

A warm ``claude`` process is spawned once, so everything in its environment is
frozen at spawn -- including ``ANTHROPIC_BASE_URL``, its auth token and
``ANTHROPIC_CUSTOM_HEADERS`` (which is how a per-turn spawn carries the turn's
GPU pool lease). Instead, each warm process is pointed at this relay, one URL
per pool slot (``http://127.0.0.1:<port>/slot/<id>``). The relay looks up which
turn currently owns that slot and, per request:

- picks the upstream that turn would have used (llm-gateway when it holds a GPU
  lease, the FCC server otherwise -- same rule as
  ``fcc_motor._build_subprocess_env``);
- replaces the process's slot secret with the turn's real credential;
- adds the CURRENT turn's ``X-Orion-Gpu-Lease`` (stripping any copy the client
  sent) and ``X-Orion-Correlation-Id``.

A ``POST /v1/messages`` with no turn bound is refused with 409, so a warm
process can never spend a model call (or a lease) outside a turn -- including
a late request from a turn that was killed for overrunning its deadline. Other
idle requests (the CLI's start-up ``/v1/models`` discovery, ``count_tokens``)
go to the default FCC upstream with the FCC credential.

Requests without the slot's secret get 401, so another local process cannot
borrow a turn's lease through the relay.

The relay runs on its own thread and event loop (uvicorn bound to 127.0.0.1),
so a slow upstream stream can never stall the governor's bus loops, and uvicorn
does not install signal handlers off the main thread.
"""

from __future__ import annotations

import contextlib
import hmac
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("orion.harness.fcc_warm_relay")

CORRELATION_HEADER = "X-Orion-Correlation-Id"

_DROP_REQUEST_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
        # Replaced per turn below.
        "authorization",
        "x-api-key",
        CORRELATION_HEADER.lower(),
    }
)
_DROP_RESPONSE_HEADERS = frozenset(
    {"content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade"}
)


@dataclass(frozen=True)
class RelayUpstream:
    """Where a request goes and with which credential."""

    base_url: str
    auth_token: str
    api_key: Optional[str] = None


@dataclass(frozen=True)
class RelayTurnBinding:
    """The turn currently using a slot. Built by the motor, per turn."""

    correlation_id: str
    upstream: RelayUpstream
    gpu_lease_header: Optional[str] = None  # already-encoded X-Orion-Gpu-Lease value


class RelayRegistry:
    """Slot secrets and current turn bindings. Shared by the governor loop and the relay thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._secrets: Dict[str, str] = {}
        self._bindings: Dict[str, RelayTurnBinding] = {}

    def register_slot(self, slot_id: str) -> str:
        """New secret for a (re)spawned slot process. Any old binding is dropped."""
        secret = secrets.token_urlsafe(32)
        with self._lock:
            self._secrets[slot_id] = secret
            self._bindings.pop(slot_id, None)
        return secret

    def forget_slot(self, slot_id: str) -> None:
        with self._lock:
            self._secrets.pop(slot_id, None)
            self._bindings.pop(slot_id, None)

    def bind(self, slot_id: str, binding: RelayTurnBinding) -> None:
        with self._lock:
            if slot_id not in self._secrets:
                raise KeyError(f"unknown warm slot {slot_id!r}")
            self._bindings[slot_id] = binding

    def unbind(self, slot_id: str) -> None:
        with self._lock:
            self._bindings.pop(slot_id, None)

    def binding(self, slot_id: str) -> Optional[RelayTurnBinding]:
        with self._lock:
            return self._bindings.get(slot_id)

    def resolve(self, slot_id: str, presented: str) -> Tuple[bool, Optional[RelayTurnBinding]]:
        with self._lock:
            expected = self._secrets.get(slot_id)
            binding = self._bindings.get(slot_id)
        if not expected or not presented or not hmac.compare_digest(expected, presented):
            return False, None
        return True, binding


def _presented_secret(headers: Any) -> str:
    auth = str(headers.get("authorization") or "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return str(headers.get("x-api-key") or "").strip()


def _json_error(status: int, message: str, *, error_type: str = "api_error"):
    from starlette.responses import JSONResponse

    return JSONResponse({"type": "error", "error": {"type": error_type, "message": message}}, status_code=status)


def build_relay_app(registry: RelayRegistry, default_upstream: RelayUpstream, *, upstream_timeout_sec: float = 10.0):
    """ASGI app. Built lazily so importing this module does not require starlette/httpx."""
    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    from orion.llm.resource_lease import GPU_LEASE_HEADER

    lease_header_lower = GPU_LEASE_HEADER.lower()
    state: Dict[str, Any] = {}

    @contextlib.asynccontextmanager
    async def _lifespan(_app):
        # No read timeout: a long generation is bounded by the motor's own
        # deadline/stall kill, which closes this connection from the client side.
        state["client"] = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=float(upstream_timeout_sec)),
            follow_redirects=False,
        )
        try:
            yield
        finally:
            client = state.pop("client", None)
            if client is not None:
                await client.aclose()

    async def relay(request: Request):
        slot_id = request.path_params["slot_id"]
        rest = request.path_params.get("path") or ""
        ok, binding = registry.resolve(slot_id, _presented_secret(request.headers))
        if not ok:
            return _json_error(401, "warm relay: bad or missing slot secret", error_type="authentication_error")
        is_messages = request.method == "POST" and rest.rstrip("/") == "v1/messages"
        if binding is None and is_messages:
            logger.warning("fcc_warm_relay_refused_unbound slot=%s path=/%s", slot_id, rest)
            # 409 invalid_request_error, not 503/overloaded: the CLI must not retry
            # its way into a later turn's binding.
            return _json_error(409, "warm relay: no chat turn is bound to this slot", error_type="invalid_request_error")
        upstream = binding.upstream if binding is not None else default_upstream
        url = upstream.base_url.rstrip("/") + "/" + rest
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _DROP_REQUEST_HEADERS and k.lower() != lease_header_lower
        }
        headers["authorization"] = f"Bearer {upstream.auth_token}"
        if upstream.api_key:
            headers["x-api-key"] = upstream.api_key
        corr = "-"
        if binding is not None:
            corr = binding.correlation_id
            headers[CORRELATION_HEADER] = binding.correlation_id
            if binding.gpu_lease_header:
                headers[GPU_LEASE_HEADER] = binding.gpu_lease_header
        body = await request.body()
        client: httpx.AsyncClient = state["client"]
        started = time.monotonic()
        try:
            upstream_req = client.build_request(request.method, url, headers=headers, content=body)
            resp = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            logger.warning(
                "fcc_warm_relay_upstream_error corr=%s slot=%s path=/%s upstream=%s error=%s",
                corr, slot_id, rest, upstream.base_url, exc,
            )
            return _json_error(502, f"warm relay: upstream unreachable: {exc}")
        logger.info(
            "fcc_warm_relay_request corr=%s slot=%s method=%s path=/%s status=%s upstream=%s lease=%s headers_ms=%d",
            corr, slot_id, request.method, rest, resp.status_code, upstream.base_url,
            bool(binding is not None and binding.gpu_lease_header),
            int((time.monotonic() - started) * 1000),
        )
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}
        async def body():
            # Close upstream on every exit, including the client (a killed warm
            # process) disconnecting, so a GPU generation never outlives its turn.
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(body(), status_code=resp.status_code, headers=out_headers)

    methods = ["GET", "POST", "HEAD", "PUT", "DELETE", "PATCH", "OPTIONS"]
    return Starlette(
        routes=[
            Route("/slot/{slot_id}/{path:path}", relay, methods=methods),
            Route("/slot/{slot_id}", relay, methods=methods),
        ],
        lifespan=_lifespan,
    )


class RelayServer:
    """uvicorn on a dedicated thread, bound to 127.0.0.1 only."""

    def __init__(
        self,
        registry: RelayRegistry,
        default_upstream: RelayUpstream,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.registry = registry
        self.default_upstream = default_upstream
        self.host = host
        self.port = int(port)
        self._server: Any = None
        self._thread: Optional[threading.Thread] = None

    def slot_base_url(self, slot_id: str) -> str:
        return f"http://{self.host}:{self.port}/slot/{slot_id}"

    def start(self, *, timeout_sec: float = 10.0) -> None:
        import uvicorn

        app = build_relay_app(self.registry, self.default_upstream)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            # None: do NOT let uvicorn reconfigure the governor's logging.
            log_config=None,
            access_log=False,
            lifespan="on",
            loop="asyncio",
            ws="none",
        )
        server = uvicorn.Server(config)
        self._server = server
        thread = threading.Thread(target=server.run, name="fcc-warm-relay", daemon=True)
        thread.start()
        self._thread = thread
        deadline = time.monotonic() + timeout_sec
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("fcc warm relay thread exited during start-up")
            if time.monotonic() > deadline:
                raise RuntimeError("fcc warm relay did not start in time")
            time.sleep(0.01)
        if self.port == 0:
            for srv in getattr(server, "servers", []) or []:
                for sock in getattr(srv, "sockets", []) or []:
                    self.port = int(sock.getsockname()[1])
                    break
        logger.info("fcc_warm_relay_started host=%s port=%s", self.host, self.port)

    def stop(self, *, timeout_sec: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
        self._server = None
        self._thread = None


def summarize_binding(binding: Optional[RelayTurnBinding]) -> Dict[str, Any]:
    """Debug view without the credential."""
    if binding is None:
        return {"bound": False}
    return {
        "bound": True,
        "correlation_id": binding.correlation_id,
        "upstream": binding.upstream.base_url,
        "lease": bool(binding.gpu_lease_header),
    }


__all__ = [
    "CORRELATION_HEADER",
    "RelayRegistry",
    "RelayServer",
    "RelayTurnBinding",
    "RelayUpstream",
    "build_relay_app",
    "summarize_binding",
]
