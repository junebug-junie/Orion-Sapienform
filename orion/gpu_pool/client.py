"""The only GPU pool client. Every caller (gateway, durable runs, world-model, diffusion) uses it.

    async with gpu_lease(bus, work_class="metacog", holder="orion-llm-gateway",
                         priority="system", deadline_sec=30, turn_correlation_id=corr) as lease:
        call(lease.grant.url)          # the card the pool chose, with its discovered model
        if lease.recalled.is_set(): ...  # an owner wants it back: finish up, take no new work

On exit the lease is released with ``ok``, or ``upstream_error`` if the block raised. A caller
always gets a grant or a typed exception (``LeaseUnavailable`` / ``LeaseBacklogged``); never an
unexplained timeout.

Transport rule (spec, "Transport-metric and reader impacts" item 1): the lease RPC travels on a
FRESH correlation id, with the turn's id carried as ``turn_correlation_id``. Waiting in line is
not transport, so it must never enter the turn's bus-synaptic causal chain.
"""
from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.gpu_pool import (
    GPU_LEASE_REQUEST_KIND, GPU_POOL_EVENT_CHANNEL, GPU_POOL_LEASE_REPLY_PREFIX, GPU_POOL_LEASE_REQUEST_CHANNEL, GpuLeaseGrantV1,
    GpuLeaseReplyV1, GpuLeaseRequestV1,
)

REPLY_PREFIX = GPU_POOL_LEASE_REPLY_PREFIX
RPC_HEALTH_LABEL = "gpu_pool_lease"   # the RPC itself answers at once: real transport, not excluded
# The pool records queue wait under this label; equilibrium excludes it (waiting is not transport).
WAIT_HOP_LABEL = "gpu_pool_wait"


class LeaseUnavailable(RuntimeError):
    def __init__(self, reason: str, lease_id: str | None = None):
        super().__init__(reason)
        self.reason, self.lease_id = reason, lease_id


class LeaseBacklogged(LeaseUnavailable):
    """Nothing can serve this class right now. Only raised for ``retryable=True`` callers: the pool
    keeps the lease and re-grants it later, and the caller is expected to come back by lease_id."""


@dataclass
class Lease:
    lease_id: str
    grant: GpuLeaseGrantV1
    recalled: asyncio.Event = field(default_factory=asyncio.Event)
    recall_by: datetime | None = None
    # Set when the pool no longer considers this lease held (expired, aborted, gone): stop using
    # the GPU now. ``recalled`` is set too, so callers watching only that still stop.
    lost: asyncio.Event = field(default_factory=asyncio.Event)


async def lease_rpc(bus: Any, req: GpuLeaseRequestV1, *, source: str, timeout_sec: float = 10.0) -> GpuLeaseReplyV1:
    env = BaseEnvelope(kind=GPU_LEASE_REQUEST_KIND, source=ServiceRef(name=source), correlation_id=uuid.uuid4(),
                       payload=req.model_dump(mode="json"))
    raw = await bus.rpc_request(GPU_POOL_LEASE_REQUEST_CHANNEL, env,
                                reply_channel=f"{REPLY_PREFIX}{uuid.uuid4().hex}",
                                timeout_sec=timeout_sec, health_label=RPC_HEALTH_LABEL)
    decoded = bus.codec.decode(raw["data"])  # rpc_request returns the raw pubsub message
    return GpuLeaseReplyV1.model_validate(decoded.envelope.payload)


@contextlib.asynccontextmanager
async def gpu_lease(
    bus: Any, *, work_class: str, holder: str, priority: str = "system", kind: str = "request",
    deadline_sec: float = 60.0, min_ctx_tokens: int = 0, turn_correlation_id: str | None = None,
    request_id: str | None = None, heartbeat_sec: float | None = None, retryable: bool = False,
) -> AsyncIterator[Lease]:
    source = holder
    deadline = datetime.now(timezone.utc) + timedelta(seconds=deadline_sec)
    req = GpuLeaseRequestV1(verb="acquire", request_id=request_id or uuid.uuid4().hex, holder=holder,
                            work_class=work_class, priority=priority, kind=kind, min_ctx_tokens=min_ctx_tokens,
                            deadline_at=deadline, turn_correlation_id=turn_correlation_id, retryable=retryable)
    lease: Lease | None = None
    reply: GpuLeaseReplyV1 | None = None
    try:
        async with bus.subscribe(GPU_POOL_EVENT_CHANNEL) as pubsub:  # before acquire: no missed grant
            reply = await lease_rpc(bus, req, source=source)
            if reply.status == "granted" and reply.grant:
                lease = Lease(reply.lease_id, reply.grant)
            elif reply.status == "queued":
                lease = await _wait_for_grant(bus, pubsub, reply.lease_id, deadline)
    except BaseException:
        # Cancelled or failed mid-acquire: withdraw by request_id (idempotent; works even if the
        # acquire reply never arrived) so nothing is later granted to a caller that has left.
        await _withdraw(bus, req, reply, source)
        raise
    if lease is None:
        if reply.status == "backlogged" and retryable:
            raise LeaseBacklogged(reply.reason or "backlogged", reply.lease_id)
        await _withdraw(bus, req, reply, source)
        raise LeaseUnavailable(reply.reason or ("deadline" if reply.status == "queued" else reply.status),
                               reply.lease_id)

    beat = asyncio.create_task(_heartbeat(bus, lease, source, heartbeat_sec or (10.0 if kind == "request" else 30.0)))
    outcome, detail = "ok", None
    try:
        yield lease
    except BaseException as exc:
        outcome, detail = "upstream_error", f"{type(exc).__name__}: {exc}"[:2000]
        raise
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
        await _quiet(lease_rpc(bus, GpuLeaseRequestV1(verb="release", lease_id=lease.lease_id,
                                                      outcome=outcome, detail=detail), source=source))


async def _wait_for_grant(bus: Any, pubsub: Any, lease_id: str, deadline: datetime) -> Lease | None:
    async def listen() -> Lease | None:
        async for msg in bus.iter_messages(pubsub):
            payload = bus.codec.decode(msg["data"]).envelope.payload or {}
            if payload.get("lease_id") != lease_id:
                continue
            if payload.get("event") == "granted" and (payload.get("detail") or {}).get("grant"):
                return Lease(lease_id, GpuLeaseGrantV1.model_validate(payload["detail"]["grant"]))
            if payload.get("event") in ("unavailable", "backlogged", "cancelled", "dead_lettered"):
                return None
        return None

    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    try:
        return await asyncio.wait_for(listen(), timeout=max(0.0, remaining))
    except asyncio.TimeoutError:
        return None


async def _heartbeat(bus: Any, lease: Lease, source: str, every: float) -> None:
    while True:
        await asyncio.sleep(every)
        try:
            reply = await lease_rpc(bus, GpuLeaseRequestV1(verb="heartbeat", lease_id=lease.lease_id), source=source)
        except Exception:  # noqa: BLE001 -- a missed beat is survivable; the TTL is 3x this
            continue
        if reply.status == "recall":
            lease.recall_by = reply.recall_by
            lease.recalled.set()
        elif reply.status != "granted":
            lease.lost.set()
            lease.recalled.set()
            return


async def _withdraw(bus: Any, req: GpuLeaseRequestV1, reply: GpuLeaseReplyV1 | None, source: str) -> None:
    lease_id = reply.lease_id if reply is not None else None
    if lease_id is None:
        # The acquire may have landed without us seeing the reply: re-acquire is idempotent on
        # request_id and returns the lease_id, which we then cancel.
        try:
            lease_id = (await lease_rpc(bus, req, source=source)).lease_id
        except BaseException:  # noqa: BLE001
            return
    if lease_id:
        await _quiet(asyncio.shield(lease_rpc(bus, GpuLeaseRequestV1(verb="cancel", lease_id=lease_id), source=source)))


async def _quiet(coro) -> None:
    with contextlib.suppress(Exception):
        await coro
