"""Leased model calls for evals: take a GPU pool seat, call it only through the gateway.

Phase 1 of docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md (approved
2026-11-12). Before this, replay evals POSTed straight to a model server's ``/v1/chat/completions``.
The pool never saw them, so it gave the same seat to Orion's own work at the same time.

Now an eval does this::

    with operator_lane_sync("chat", holder="operator:promise-ledger-replay") as lane:
        reply = lane.chat(body)        # POST gateway /v1/chat/completions, model=operator_chat,
                                       # X-Orion-Gpu-Lease=<the hold>

What that means:

* ONE pool hold per run per lane, work class ``operator_<lane>`` (config/gpu_pool.yaml), priority
  ``background``, heartbeated. It waits in the pool's own queue behind every one of Orion's own
  background leases (GPU_POOL_OPERATOR_LAST). It never lands on chat's card or hecate unless Juniper
  has lent the card. The heat reflex sheds it on the same lines as Orion's background work.
* Every call goes to orion-llm-gateway with the hold's ref, so the gateway attaches it as a child of
  the hold (pool verb ``attach``). There is no direct-URL path and no fallback to one.
* When an owner needs the seat, the pool recalls the hold. The next call (or a call the gateway
  refuses because the hold is gone) raises ``LaneRecalled``. The eval writes a partial report and
  exits. Nothing resumes on its own (Juniper, answer 6).
* The hold is released on every exit path: normal end, exception, Ctrl+C (the pool's heartbeat TTL
  ends it if the process is killed; it is not retryable, so it never re-queues for nobody).

The pool records the hold under the eval's ``holder`` and the class ``operator_*``, so
``gpu_pool_events`` shows every eval run: ``work_class like 'operator%'``.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional

import httpx

OPERATOR_PRIORITY = "background"
# Lane name an eval asks for -> the pool's operator work class (= the gateway route of the same name).
OPERATOR_LANES: dict[str, str] = {
    "any": "operator",
    "chat": "operator_chat",
    "agent": "operator_agent",
    "agent-gpu2": "operator_agent_gpu2",
    "agent-deep": "operator_agent_deep",
    "metacog": "operator_metacog",
    "fast": "operator_fast",
}
DEFAULT_GATEWAY_URL = "http://127.0.0.1:8210"
GATEWAY_URL_ENV = "ORION_LLM_GATEWAY_URL"
BUS_URL_ENV = "ORION_BUS_URL"
# How long a run waits in the pool's queue for its seat before giving up. Operator work is last in
# line by design, so this is long; the eval says how long it waited when it gives up.
DEFAULT_QUEUE_WAIT_SEC = 6 * 3600.0
HEARTBEAT_SEC = 30.0   # the pool's hold TTL is 90 s (config/gpu_pool.yaml hold_lease_ttl_sec)
DEFAULT_CALL_TIMEOUT_SEC = 900.0
# Gateway refusals that mean the hold is gone (recalled past its grace, expired, re-granted).
_HOLD_GONE_MARKERS = ("gpu_pool_recalled", "hold_not_granted", "hold_unknown", "stale_hold_generation",
                      "not_a_hold")
WAITING_NOTICE_SEC = 10.0   # say once that the run is waiting for its seat, after this long
EXIT_RECALLED = 3   # exit status an eval uses after writing its partial report


class LaneRecalled(RuntimeError):
    """The pool took the seat back (an owner waits, or the hold is gone). Write a partial report and
    stop; do not retry."""


class LaneUnavailable(RuntimeError):
    """The pool refused the hold or the queue wait ran out. Nothing was called."""


class LaneCallError(RuntimeError):
    """One gateway call failed for a reason that is not a recall (bad request, upstream error,
    transport). ``status`` is the HTTP status, None for a transport error."""

    def __init__(self, message: str, status: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status, self.body = status, body


def operator_class(lane: str) -> str:
    try:
        return OPERATOR_LANES[lane]
    except KeyError:
        raise ValueError(f"unknown operator lane {lane!r}; one of {sorted(OPERATOR_LANES)}") from None


def gateway_url(url: Optional[str] = None) -> str:
    return (url or os.environ.get(GATEWAY_URL_ENV) or DEFAULT_GATEWAY_URL).rstrip("/")


def _hold_gone(body: Any) -> bool:
    text = str(body) if not isinstance(body, (bytes, bytearray)) else body.decode("utf-8", "replace")
    return any(m in text for m in _HOLD_GONE_MARKERS)


@dataclass
class OperatorLane:
    """A granted operator hold on one lane, plus the only way to call that lane."""

    lane: str
    route: str
    holder: str
    gateway: str
    lease: Any                     # orion.gpu_pool.client.Lease
    calls: int = 0
    recalled_reason: Optional[str] = None
    transport: Any = field(default=None, repr=False)   # httpx transport (tests only)
    _client: Optional[httpx.Client] = field(default=None, repr=False)

    @property
    def lease_id(self) -> str:
        return self.lease.lease_id

    @property
    def role(self) -> str:
        return self.lease.grant.role

    def ref(self) -> Any:
        from orion.schemas.gpu_pool import GpuLeaseRefV1

        g = self.lease.grant
        return GpuLeaseRefV1(lease_id=g.lease_id, generation=g.generation, role=g.role, holder=self.holder)

    def headers(self) -> dict[str, str]:
        from orion.llm.resource_lease import GPU_LEASE_HEADER, encode_gpu_lease_header

        return {"Content-Type": "application/json", GPU_LEASE_HEADER: encode_gpu_lease_header(self.ref())}

    def summary(self) -> dict[str, Any]:
        """What an eval report records about its seat."""
        return {"lane": self.lane, "work_class": self.route, "holder": self.holder, "lease_id": self.lease_id,
                "role": self.role, "priority": OPERATOR_PRIORITY, "calls": self.calls,
                "recalled": self.recalled_reason}

    def check(self) -> None:
        """Raise LaneRecalled once the pool has asked for the seat back (or lost the hold)."""
        if self.recalled_reason is None and self.lease.recalled.is_set():
            self.recalled_reason = "lost" if self.lease.lost.is_set() else "recalled"
        if self.recalled_reason is not None:
            self.lease.release_outcome = "cancelled"
            self.lease.release_detail = f"operator_lane_{self.recalled_reason}"
            raise LaneRecalled(f"{self.lane}: the GPU pool took the seat back ({self.recalled_reason}); "
                               f"lease {self.lease_id}")

    def _body(self, body: dict[str, Any]) -> dict[str, Any]:
        return {**body, "model": self.route}

    def _result(self, status: int, payload: Any, text: str) -> dict[str, Any]:
        if status == 200 and isinstance(payload, dict):
            return payload
        if _hold_gone(payload if payload is not None else text):
            self.recalled_reason = "hold_gone"
            self.check()
        raise LaneCallError(f"http {status} {text[:200]}", status, payload)

    # -- sync (the replay evals are plain scripts) ------------------------------------------------
    def post(self, path: str, body: dict[str, Any], timeout_sec: float = DEFAULT_CALL_TIMEOUT_SEC) -> dict[str, Any]:
        """POST ``body`` to the gateway ``path`` under this hold. Never any other URL."""
        self.check()
        if self._client is None:
            self._client = httpx.Client(transport=self.transport) if self.transport is not None else httpx.Client()
        self.calls += 1
        try:
            r = self._client.post(f"{self.gateway}{path}", json=self._body(body), headers=self.headers(),
                                  timeout=timeout_sec)
        except httpx.HTTPError as exc:
            self.check()
            raise LaneCallError(f"transport {type(exc).__name__}: {exc}") from exc
        try:
            payload = r.json()
        except ValueError:
            payload = None
        return self._result(r.status_code, payload, r.text)

    def chat(self, body: dict[str, Any], timeout_sec: float = DEFAULT_CALL_TIMEOUT_SEC) -> dict[str, Any]:
        """OpenAI chat completions through the gateway."""
        return self.post("/v1/chat/completions", body, timeout_sec)

    def messages(self, body: dict[str, Any], timeout_sec: float = DEFAULT_CALL_TIMEOUT_SEC) -> dict[str, Any]:
        """Anthropic messages through the gateway."""
        return self.post("/v1/messages", body, timeout_sec)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


async def _connect_bus(bus_url: Optional[str]) -> Any:
    from orion.core.bus.async_service import OrionBusAsync

    url = bus_url or os.environ.get(BUS_URL_ENV)
    if not url:
        raise LaneUnavailable(f"{BUS_URL_ENV} is not set: the eval cannot ask the GPU pool for a seat")
    bus = OrionBusAsync(url=url)
    await bus.connect()
    return bus


@contextlib.asynccontextmanager
async def operator_lane(lane: str, *, holder: str, bus: Any = None, bus_url: Optional[str] = None,
                        gateway: Optional[str] = None, queue_wait_sec: float = DEFAULT_QUEUE_WAIT_SEC,
                        heartbeat_sec: float = HEARTBEAT_SEC, http_transport: Any = None
                        ) -> AsyncIterator[OperatorLane]:
    """Hold an operator seat on ``lane`` for the block. ``holder`` should be ``operator:<eval-name>``."""
    from orion.gpu_pool.client import LeaseUnavailable, gpu_lease

    if not holder.startswith("operator:"):
        raise ValueError(f"holder must start with 'operator:' (got {holder!r})")
    route = operator_class(lane)
    own_bus = bus is None
    if own_bus:
        bus = await _connect_bus(bus_url)
    try:
        try:
            cm = gpu_lease(bus, work_class=route, holder=holder, priority=OPERATOR_PRIORITY, kind="hold",
                           deadline_sec=queue_wait_sec, heartbeat_sec=heartbeat_sec,
                           request_id=f"{holder}:{uuid.uuid4().hex}")
            lease = await cm.__aenter__()
        except LeaseUnavailable as exc:
            raise LaneUnavailable(f"{lane}: the GPU pool did not grant {route} ({exc.reason})") from exc
        handle = OperatorLane(lane=lane, route=route, holder=holder, gateway=gateway_url(gateway), lease=lease,
                              transport=http_transport)
        exc_info: tuple = (None, None, None)
        try:
            yield handle
        except BaseException as exc:  # noqa: BLE001 -- re-raised below after the release
            exc_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            handle.close()
            if isinstance(exc_info[1], LaneRecalled) and lease.release_outcome is None:
                lease.release_outcome = "cancelled"
            await cm.__aexit__(*exc_info)
    finally:
        if own_bus:
            with contextlib.suppress(Exception):
                await bus.close()


_ready_event = threading.Event   # test seam: the caller's wait for its seat


@contextlib.contextmanager
def operator_lane_sync(lane: str, *, holder: str, bus_url: Optional[str] = None, gateway: Optional[str] = None,
                       queue_wait_sec: float = DEFAULT_QUEUE_WAIT_SEC, heartbeat_sec: float = HEARTBEAT_SEC,
                       http_transport: Any = None) -> Iterator[OperatorLane]:
    """``operator_lane`` for a plain (non-async) script: the hold and its heartbeat live on a
    background event loop thread; calls are made from the caller's thread."""
    ready = _ready_event()
    box: dict[str, Any] = {}
    loop = asyncio.new_event_loop()

    async def keep() -> None:
        box["task"] = asyncio.current_task()
        box["stop"] = asyncio.Event()
        try:
            async with operator_lane(lane, holder=holder, bus_url=bus_url, gateway=gateway,
                                     queue_wait_sec=queue_wait_sec, heartbeat_sec=heartbeat_sec,
                                     http_transport=http_transport) as handle:
                box["lane"] = handle
                ready.set()
                await box["stop"].wait()
                if box.get("raise_recalled"):
                    raise LaneRecalled(box["raise_recalled"])
        except LaneRecalled:
            pass   # the caller already has it; this only marks the release "cancelled"
        except BaseException as exc:  # noqa: BLE001 -- handed to the caller's thread
            box["error"] = exc
        finally:
            ready.set()

    thread = threading.Thread(target=lambda: loop.run_until_complete(keep()), name=f"operator-lane-{lane}",
                              daemon=True)
    thread.start()
    try:
        if not ready.wait(WAITING_NOTICE_SEC):
            import sys

            print(f"operator lane {lane}: waiting in the GPU pool's queue for {operator_class(lane)} "
                  f"(behind all of Orion's own work; up to {queue_wait_sec / 3600:.1f} h)", file=sys.stderr, flush=True)
            ready.wait()
    except BaseException:
        # Ctrl+C while queued: cancel the wait so gpu_lease withdraws the queued hold, then stop.
        task = box.get("task")
        if task is not None:
            loop.call_soon_threadsafe(task.cancel)
        thread.join(timeout=15.0)
        if not thread.is_alive():
            loop.close()
        raise
    if "lane" not in box:
        thread.join()
        loop.close()
        raise box.get("error") or LaneUnavailable(f"{lane}: no seat")
    handle: OperatorLane = box["lane"]
    try:
        yield handle
    except LaneRecalled as exc:
        box["raise_recalled"] = str(exc)
        raise
    except BaseException as exc:
        # The eval failed in this thread: upstream_error, like the async form; a Ctrl+C is cancelled.
        if handle.lease.release_outcome is None:
            interrupted = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            handle.lease.release_outcome = "cancelled" if interrupted else "upstream_error"
            handle.lease.release_detail = f"{type(exc).__name__}: {exc}"[:2000]
        raise
    finally:
        loop.call_soon_threadsafe(box["stop"].set)
        thread.join()
        loop.close()
        if "error" in box and box["error"] is not None and not isinstance(box["error"], LaneRecalled):
            # The release itself failed; the eval's own outcome stands, but say so.
            import logging

            logging.getLogger(__name__).warning("operator_lane_release_failed lane=%s lease=%s err=%r",
                                                lane, handle.lease_id, box["error"])


class OperatorLanes:
    """Several lanes for one run (``with OperatorLanes(holder=...) as lanes: lanes.get("agent")``).
    Each lane's hold is taken on its first use and kept for the run: one hold per run per lane. A lane
    the run never uses is never asked for (chat's card waits until Juniper lends it, so an eval that
    does not need it must not block on it). Every hold is released when the block ends."""

    def __init__(self, *, holder: str, **lane_kw: Any):
        self.holder, self.lane_kw = holder, lane_kw
        self._stack = contextlib.ExitStack()
        self._lanes: dict[str, OperatorLane] = {}

    def __enter__(self) -> "OperatorLanes":
        self._stack.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._stack.__exit__(*exc)

    def get(self, lane: str) -> OperatorLane:
        if lane not in self._lanes:
            self._lanes[lane] = self._stack.enter_context(
                operator_lane_sync(lane, holder=self.holder, **self.lane_kw))
        return self._lanes[lane]

    def summary(self) -> list[dict[str, Any]]:
        return [h.summary() for h in self._lanes.values()]


__all__ = ["DEFAULT_GATEWAY_URL", "EXIT_RECALLED", "LaneCallError", "LaneRecalled", "LaneUnavailable",
           "OPERATOR_LANES", "OPERATOR_PRIORITY", "OperatorLane", "OperatorLanes", "gateway_url", "operator_class",
           "operator_lane", "operator_lane_sync"]
