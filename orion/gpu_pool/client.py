"""The only GPU pool client. Every caller (gateway, durable runs, world-model, diffusion) uses it.

    async with gpu_lease(bus, work_class="metacog", holder="orion-llm-gateway",
                         priority="system", deadline_sec=30, turn_correlation_id=corr) as lease:
        call(lease.grant.url)          # the card the pool chose, with its discovered model
        if lease.recalled.is_set(): ...  # an owner wants it back: finish up, take no new work

On exit the lease is released with ``ok``, or ``upstream_error`` if the block raised. A caller
always gets a grant or a typed exception (``LeaseUnavailable`` / ``LeaseBacklogged``); never an
unexplained timeout.

Stage 4 (durable runs on pool holds, docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md):

    reply = await acquire_hold(bus, holder="durable-runs:<run_id>", work_class="agent",
                               request_id=f"{run_id}:{attempt}", priority="background")
    # granted -> GpuLeaseRefV1 via hold_ref(reply); queued/backlogged -> interrupt, wake on the
    # pool's "granted" event for this lease_id; unavailable -> fail the run with reply.reason
    reply = await heartbeat_lease(bus, lease_id, source=...)   # "recall" -> release at next node
    reply = await lease_status(bus, lease_id, source=...)      # resume: granted / queued / unknown_lease
    await release_lease(bus, lease_id, source=..., outcome="ok")

    async with attached_lease(bus, ref, work_class="agent", holder="orion-llm-gateway",
                              deadline_sec=60, turn_correlation_id=corr) as lease:
        call(lease.grant.url)   # one LLM call in the hold's slot; never a second lease

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
    GpuLeaseRefV1, GpuLeaseReplyV1, GpuLeaseRequestV1,
)

REPLY_PREFIX = GPU_POOL_LEASE_REPLY_PREFIX
# One lease RPC never waits longer than this (and never longer than the caller's own deadline):
# the pool answers acquire at once (queued/granted), so a silent pool is an unreachable pool.
LEASE_RPC_TIMEOUT_SEC = 10.0
LEASE_RPC_MIN_TIMEOUT_SEC = 1.0
# Withdrawing a failed acquire is best effort; it must not add another full RPC wait on top.
WITHDRAW_RPC_TIMEOUT_SEC = 2.0
RPC_HEALTH_LABEL = "gpu_pool_lease"   # the RPC itself answers at once: real transport, not excluded
# The pool records queue wait under this label; equilibrium excludes it (waiting is not transport).
WAIT_HOP_LABEL = "gpu_pool_wait"


class LeaseUnavailable(RuntimeError):
    def __init__(self, reason: str, lease_id: str | None = None):
        super().__init__(reason)
        self.reason, self.lease_id = reason, lease_id


class PoolRpcTimeout(asyncio.TimeoutError):
    """The acquire RPC went unanswered. ``full`` is True only when the wait was the full
    LEASE_RPC_TIMEOUT_SEC: a timeout shortened by a caller's tiny deadline says nothing about the
    pool's health, so callers must not treat it as "pool down"."""

    def __init__(self, full: bool):
        super().__init__("gpu pool acquire RPC timed out")
        self.full = full


_BACKGROUND: set = set()


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
    # A caller that stopped work on purpose (e.g. aborted its upstream call because the lease was
    # recalled or lost) sets this so the release reports that outcome instead of ok/upstream_error.
    release_outcome: str | None = None
    release_detail: str | None = None


async def lease_rpc(bus: Any, req: GpuLeaseRequestV1, *, source: str, timeout_sec: float = 10.0) -> GpuLeaseReplyV1:
    reply_channel = f"{REPLY_PREFIX}{uuid.uuid4().hex}"
    # rpc_request only listens on reply_channel; the responder (Rabbit) replies to env.reply_to and
    # stays silent without it. Both must name the same channel.
    env = BaseEnvelope(kind=GPU_LEASE_REQUEST_KIND, source=ServiceRef(name=source), correlation_id=uuid.uuid4(),
                       reply_to=reply_channel, payload=req.model_dump(mode="json"))
    raw = await bus.rpc_request(GPU_POOL_LEASE_REQUEST_CHANNEL, env, reply_channel=reply_channel,
                                timeout_sec=timeout_sec, health_label=RPC_HEALTH_LABEL)
    decoded = bus.codec.decode(raw["data"])  # rpc_request returns the raw pubsub message
    return GpuLeaseReplyV1.model_validate(decoded.envelope.payload)


@contextlib.asynccontextmanager
async def gpu_lease(
    bus: Any, *, work_class: str, holder: str, priority: str = "system", kind: str = "request",
    deadline_sec: float = 60.0, min_ctx_tokens: int = 0, turn_correlation_id: str | None = None,
    request_id: str | None = None, heartbeat_sec: float | None = None, retryable: bool = False,
) -> AsyncIterator[Lease]:
    deadline = datetime.now(timezone.utc) + timedelta(seconds=deadline_sec)
    req = GpuLeaseRequestV1(verb="acquire", request_id=request_id or uuid.uuid4().hex, holder=holder,
                            work_class=work_class, priority=priority, kind=kind, min_ctx_tokens=min_ctx_tokens,
                            deadline_at=deadline, turn_correlation_id=turn_correlation_id, retryable=retryable)
    async with _held(bus, req, source=holder, deadline=deadline, deadline_sec=deadline_sec,
                     heartbeat_sec=heartbeat_sec or (10.0 if kind == "request" else 30.0),
                     retryable=retryable) as lease:
        yield lease


@contextlib.asynccontextmanager
async def attached_lease(
    bus: Any, ref: GpuLeaseRefV1, *, work_class: str, holder: str, priority: str = "background",
    deadline_sec: float = 60.0, turn_correlation_id: str | None = None, request_id: str | None = None,
    heartbeat_sec: float = 10.0,
) -> AsyncIterator[Lease]:
    """One call under a durable run's hold (``verb=attach``). Same contract as ``gpu_lease``: a grant
    in the hold's slot, or ``LeaseUnavailable`` -- e.g. ``stale_hold_generation:<n>`` /
    ``hold_not_granted:<status>`` when the run's hold moved or ended, ``hold_unknown``. It never
    takes a lease of its own: a call whose run holds the role's only slot would queue behind
    itself forever (spec, "Why the gateway must attach, not lease")."""
    deadline = datetime.now(timezone.utc) + timedelta(seconds=deadline_sec)
    req = GpuLeaseRequestV1(verb="attach", request_id=request_id or uuid.uuid4().hex, holder=holder,
                            work_class=work_class, priority=priority, deadline_at=deadline,
                            turn_correlation_id=turn_correlation_id, hold_lease_id=ref.lease_id,
                            hold_generation=ref.generation)
    async with _held(bus, req, source=holder, deadline=deadline, deadline_sec=deadline_sec,
                     heartbeat_sec=heartbeat_sec, retryable=False) as lease:
        yield lease


async def acquire_hold(
    bus: Any, *, holder: str, work_class: str, request_id: str, priority: str = "background",
    deadline_at: datetime | None = None, min_ctx_tokens: int = 0, turn_correlation_id: str | None = None,
    source: str | None = None, timeout_sec: float = LEASE_RPC_TIMEOUT_SEC,
) -> GpuLeaseReplyV1:
    """Ask for a durable run's hold; returns at once (granted / queued / backlogged / unavailable).
    Idempotent on ``request_id`` (``<run_id>:<attempt>``): re-asking after a lost reply returns the
    same lease. Always ``retryable``: an expired or aborted hold re-queues with the same lease_id."""
    req = GpuLeaseRequestV1(verb="acquire", request_id=request_id, holder=holder, work_class=work_class,
                            priority=priority, kind="hold", min_ctx_tokens=min_ctx_tokens, deadline_at=deadline_at,
                            turn_correlation_id=turn_correlation_id, retryable=True)
    return await lease_rpc(bus, req, source=source or holder, timeout_sec=timeout_sec)


async def heartbeat_lease(bus: Any, lease_id: str, *, source: str,
                          timeout_sec: float = LEASE_RPC_TIMEOUT_SEC) -> GpuLeaseReplyV1:
    """granted -> keep going; recall -> finish the current node and release before ``recall_by``;
    anything else -> the lease is no longer held (stop, then ``lease_status`` to see why)."""
    return await lease_rpc(bus, GpuLeaseRequestV1(verb="heartbeat", lease_id=lease_id), source=source,
                           timeout_sec=timeout_sec)


async def lease_status(bus: Any, lease_id: str, *, source: str,
                       timeout_sec: float = LEASE_RPC_TIMEOUT_SEC) -> GpuLeaseReplyV1:
    """Read-only. granted/recall -> continue; queued/backlogged -> wait again; ok -> released;
    unavailable -> ended with ``reason``; unknown_lease -> ask again with a new request_id."""
    return await lease_rpc(bus, GpuLeaseRequestV1(verb="status", lease_id=lease_id), source=source,
                           timeout_sec=timeout_sec)


async def release_lease(bus: Any, lease_id: str, *, source: str, outcome: str = "ok", detail: str | None = None,
                        timeout_sec: float = LEASE_RPC_TIMEOUT_SEC) -> GpuLeaseReplyV1:
    return await lease_rpc(bus, GpuLeaseRequestV1(verb="release", lease_id=lease_id, outcome=outcome,
                                                  detail=detail), source=source, timeout_sec=timeout_sec)


def hold_ref(reply: GpuLeaseReplyV1, holder: str) -> GpuLeaseRefV1:
    """The GpuLeaseRefV1 a granted hold's calls carry (header X-Orion-Gpu-Lease / options.gpu_lease)."""
    if reply.status not in ("granted", "recall") or reply.grant is None:
        raise LeaseUnavailable(f"hold_not_granted:{reply.status}", reply.lease_id)
    return GpuLeaseRefV1(lease_id=reply.grant.lease_id, generation=reply.grant.generation,
                         role=reply.grant.role, holder=holder)


@contextlib.asynccontextmanager
async def _held(bus: Any, req: GpuLeaseRequestV1, *, source: str, deadline: datetime, deadline_sec: float,
                heartbeat_sec: float, retryable: bool) -> AsyncIterator[Lease]:
    rpc_timeout = max(LEASE_RPC_MIN_TIMEOUT_SEC, min(LEASE_RPC_TIMEOUT_SEC, float(deadline_sec)))
    lease: Lease | None = None
    reply: GpuLeaseReplyV1 | None = None
    unreachable = False
    try:
        async with bus.subscribe(GPU_POOL_EVENT_CHANNEL) as pubsub:  # before acquire: no missed grant
            try:
                reply = await lease_rpc(bus, req, source=source, timeout_sec=rpc_timeout)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                unreachable = True
                raise PoolRpcTimeout(full=rpc_timeout >= LEASE_RPC_TIMEOUT_SEC) from exc
            wait_reason = None
            if reply.status == "granted" and reply.grant:
                lease = Lease(reply.lease_id, reply.grant)
            elif reply.status == "queued":
                lease, wait_reason = await _wait_for_grant(bus, pubsub, reply.lease_id, deadline)
    except BaseException:
        # Cancelled or failed mid-acquire: withdraw by request_id (idempotent; works even if the
        # acquire reply never arrived) so nothing is later granted to a caller that has left.
        if unreachable:
            # A slow (not dead) pool may still have admitted the acquire: withdraw in the
            # background, bounded, rather than stall this caller or leave a slot for nobody.
            task = asyncio.ensure_future(_quiet(_withdraw(bus, req, None, source)))
            _BACKGROUND.add(task)
            task.add_done_callback(_BACKGROUND.discard)
        else:
            await _withdraw(bus, req, reply, source)
        raise
    if lease is None:
        if reply.status == "backlogged" and retryable:
            raise LeaseBacklogged(reply.reason or "backlogged", reply.lease_id)
        await _withdraw(bus, req, reply, source)
        raise LeaseUnavailable(reply.reason or wait_reason
                               or ("deadline" if reply.status == "queued" else reply.status), reply.lease_id)

    beat = asyncio.create_task(_heartbeat(bus, lease, source, heartbeat_sec))
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
        if lease.release_outcome:
            outcome, detail = lease.release_outcome, lease.release_detail or detail
        await _quiet(lease_rpc(bus, GpuLeaseRequestV1(verb="release", lease_id=lease.lease_id,
                                                      outcome=outcome, detail=detail), source=source))


async def _wait_for_grant(bus: Any, pubsub: Any, lease_id: str,
                          deadline: datetime) -> tuple[Lease | None, str | None]:
    """(lease, None) on grant; (None, the pool's reason) when it refused while we waited -- e.g.
    "min_ctx_exceeds_class:65536" -- so the caller reports the real cause, not a generic deadline."""
    async def listen() -> tuple[Lease | None, str | None]:
        async for msg in bus.iter_messages(pubsub):
            payload = bus.codec.decode(msg["data"]).envelope.payload or {}
            if payload.get("lease_id") != lease_id:
                continue
            if payload.get("event") == "granted" and (payload.get("detail") or {}).get("grant"):
                return Lease(lease_id, GpuLeaseGrantV1.model_validate(payload["detail"]["grant"])), None
            if payload.get("event") in ("unavailable", "backlogged", "cancelled", "dead_lettered"):
                return None, payload.get("reason") or payload.get("event")
        return None, None

    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    try:
        return await asyncio.wait_for(listen(), timeout=max(0.0, remaining))
    except asyncio.TimeoutError:
        return None, "deadline"


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
    """Best effort, bounded to WITHDRAW_RPC_TIMEOUT_SEC per RPC. When the acquire RPC itself
    timed out, callers run it in the background instead: an unreachable pool must not cost the
    caller a second wait."""
    lease_id = reply.lease_id if reply is not None else None
    if lease_id is None:
        # The acquire may have landed without us seeing the reply: re-acquire is idempotent on
        # request_id and returns the lease_id, which we then cancel.
        try:
            lease_id = (await lease_rpc(bus, req, source=source, timeout_sec=WITHDRAW_RPC_TIMEOUT_SEC)).lease_id
        except BaseException:  # noqa: BLE001
            return
    if lease_id:
        await _quiet(asyncio.shield(lease_rpc(bus, GpuLeaseRequestV1(verb="cancel", lease_id=lease_id),
                                              source=source, timeout_sec=WITHDRAW_RPC_TIMEOUT_SEC)))


async def _quiet(coro) -> None:
    with contextlib.suppress(Exception):
        await coro
