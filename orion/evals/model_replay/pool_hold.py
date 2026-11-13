"""Operator holds that pin each model's card for one replay task, released on every exit path.

WHY OPERATOR CLASSES (leased evals, 2026-11-12,
docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md, Phase 1). A caller
cannot ask the pool for a role or a profile, only a work class (`config/gpu_pool.yaml` classes). The
replay asks for the pinned operator classes and then CHECKS what it got:

  * Q4 (gpu1):     class ``operator_agent`` -> roles [agent] only. Verified: grant.role == "agent"
                   and grant.profile_name is the Q4 profile.
  * Bonsai (gpu2): class ``operator_agent_gpu2`` -> roles [agent-gpu2] only. Verified the same way.

``LeaseHoldTransport`` (the replay's transport) asks at ``background`` priority through the lease
RPC, heartbeats each granted hold, and makes it non-retryable. So the replay waits behind every one
of Orion's own background leases, an owner that waits recalls it (the transport's ``recalled`` event;
the runner then stops and writes a partial report), and the heat reflex sheds it like Orion's own
background work. A killed process's hold ends on its own at the pool's heartbeat TTL.

``BusControlTransport`` (the control verb ``hold``: interactive, operator, no heartbeat) is kept only
for a hand-run reservation Juniper starts herself (scripts/gpu_pool_clear_gpu2.py with
``RESERVE_SEATS``), the same kind of hold as the Hub's experiment-seat button.

Control-verb holds have no heartbeat and no expiry (``lease_graph.py``: operator -> expiry None), so a
leaked one stays until the role's ``max_hold_sec`` or forever on gpu1. For every hold, either kind:
  1. every hold request is appended to ``holds.jsonl`` BEFORE it is sent, its lease id as soon as the
     pool answers; a hold RPC that times out (the pool may still have created the lease) triggers a
     sweep of the pool's own lease table for this holder (``sweep``);
  2. ``PoolHolds`` is an async context manager whose ``__aexit__`` releases everything, and the
     runner turns SIGTERM/SIGHUP into the same unwind;
  3. ``release_leftovers(run_dir)`` (``--release-leftovers``) releases anything a SIGKILL left.

The pool is driven only through ``ControlTransport``; tests plug in a fake one.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

Q4_PROFILE = "qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex"
BONSAI_PROFILE = "ternary-bonsai2-27b-pq2-v100-32gb-circe-agent"


@dataclass(frozen=True)
class SeatSpec:
    """What one model's hold must land on."""

    model_key: str           # "q4" | "bonsai"
    work_class: str
    expect_role: str
    expect_profile: str
    extra_slot_holds: int = 0  # additional holds on the same class to fill the seat's other slots


# The replay's seats: pinned operator classes (one role each), background, behind Orion's own work.
SEATS: dict[str, SeatSpec] = {
    "q4": SeatSpec("q4", "operator_agent", "agent", Q4_PROFILE),
    "bonsai": SeatSpec("bonsai", "operator_agent_gpu2", "agent-gpu2", BONSAI_PROFILE),
}
# A hand-run reservation (scripts/gpu_pool_clear_gpu2.py, BusControlTransport, interactive): gpu1 via
# class memory_distill, then gpu2 via class agent falling through once agent is full.
RESERVE_SEATS: dict[str, SeatSpec] = {
    "q4": SeatSpec("q4", "memory_distill", "agent", Q4_PROFILE),
    "bonsai": SeatSpec("bonsai", "agent", "agent-gpu2", BONSAI_PROFILE),
}
# gpu1 first (the reservation's class `agent` cannot then land on agent).
ACQUIRE_ORDER = ("q4", "bonsai")


class HoldRefused(RuntimeError):
    """The pool refused, timed out, or granted the wrong seat. Every hold taken so far is released."""


@dataclass
class Grant:
    lease_id: str
    role: str
    url: str
    profile_name: Optional[str]
    model_file: Optional[str]
    ctx_per_slot: Optional[int]
    cards: list[str]
    generation: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Grant":
        return cls(lease_id=str(d["lease_id"]), role=str(d.get("role")), url=str(d.get("url")),
                   profile_name=d.get("profile_name"), model_file=d.get("model_file"),
                   ctx_per_slot=d.get("ctx_per_slot"), cards=list(d.get("cards") or []),
                   generation=int(d.get("generation") or 1))


class ControlTransport(Protocol):
    async def hold(self, work_class: str, actor: str) -> dict[str, Any]:
        """verb=hold. Returns ``{"ok", "reason", "detail": GpuLeaseReplyV1-dict}``."""

    async def release(self, lease_id: str, actor: str) -> dict[str, Any]:
        """verb=release (granted) -- the pool treats an un-granted lease's release as a cancel too."""

    async def cancel(self, lease_id: str, actor: str) -> dict[str, Any]:
        """verb=cancel (queued)."""

    async def wait_granted(self, lease_id: str, timeout_sec: float) -> Optional[dict[str, Any]]:
        """The grant dict once the pool emits ``granted`` for ``lease_id``; None on timeout/refusal."""


@dataclass
class HoldLedger:
    """Append-only record of every lease this replay took and gave back, for crash recovery."""

    path: Path

    def append(self, **row: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row.setdefault("t", time.time())
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def outstanding(self) -> list[str]:
        if not self.path.exists():
            return []
        live: dict[str, bool] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            lid = row.get("lease_id")
            if not lid:
                continue
            if row.get("action") == "acquired":
                live[lid] = True
            elif row.get("action") == "released":
                live.pop(lid, None)
        return sorted(live)


@dataclass
class PoolHolds:
    """``async with PoolHolds(...) as seats:`` -> ``{"q4": Grant, "bonsai": Grant}``, all released on exit."""

    transport: ControlTransport
    ledger: HoldLedger
    models: tuple[str, ...] = ACQUIRE_ORDER
    seat_specs: dict[str, SeatSpec] = field(default_factory=lambda: dict(SEATS))
    actor: str = "bonsai-replay-eval"
    grant_timeout_sec: float = 1800.0
    # Live lease ids the pool itself holds for HOLDER (reads /v1/pool); used when a hold RPC failed
    # without telling us the lease id. None = no sweep (tests, or no pool HTTP).
    sweep: Optional[Callable[[], Any]] = None
    seats: dict[str, Grant] = field(default_factory=dict)
    _held: list[str] = field(default_factory=list)
    _unknown: bool = False

    async def __aenter__(self) -> dict[str, Grant]:
        try:
            for key in ACQUIRE_ORDER:
                if key not in self.models:
                    continue
                spec = self.seat_specs[key]
                self.seats[key] = await self._acquire_verified(spec)
                for _ in range(spec.extra_slot_holds):
                    await self._acquire_extra(spec)
        except BaseException:
            await self.release_all(reason="acquire_failed")
            raise
        return dict(self.seats)

    async def _sweep_unknown(self) -> None:
        if self.sweep is None:
            return
        try:
            ids = await self.sweep()
        except Exception:  # noqa: BLE001 -- nothing better to do; --release-leftovers sweeps again
            return
        for lid in ids:
            if lid not in self._held:
                self._held.append(lid)
                self.ledger.append(action="acquired", lease_id=lid, via="sweep")

    async def __aexit__(self, *exc: Any) -> None:
        await self.release_all(reason="task_done" if exc[0] is None else f"exit:{exc[0].__name__}")

    async def _hold(self, spec: SeatSpec) -> tuple[Optional[str], Optional[dict[str, Any]], str]:
        """(lease_id, grant-or-None, reason). Records the lease in the ledger before waiting."""
        self.ledger.append(action="requested", model=spec.model_key, work_class=spec.work_class)
        try:
            reply = await self.transport.hold(spec.work_class, self.actor)
        except BaseException:
            self._unknown = True  # the pool may have created a lease we never heard about
            raise
        detail = reply.get("detail") or {}
        lease_id = detail.get("lease_id")
        if lease_id:
            self._held.append(lease_id)
            self.ledger.append(action="acquired", lease_id=lease_id, model=spec.model_key,
                               work_class=spec.work_class, status=detail.get("status"))
        if not reply.get("ok"):
            return lease_id, None, f"refused:{reply.get('reason') or detail.get('status')}"
        if detail.get("status") == "granted" and detail.get("grant"):
            return lease_id, detail["grant"], "granted"
        grant = await self.transport.wait_granted(lease_id, self.grant_timeout_sec) if lease_id else None
        return lease_id, grant, "granted" if grant else "grant_timeout"

    async def _acquire_verified(self, spec: SeatSpec) -> Grant:
        lease_id, grant, why = await self._hold(spec)
        if grant is None:
            raise HoldRefused(f"{spec.model_key}: {why}")
        g = Grant.from_dict(grant)
        if g.role != spec.expect_role or g.profile_name != spec.expect_profile:
            raise HoldRefused(f"{spec.model_key}: wrong seat role={g.role} profile={g.profile_name} "
                              f"(want {spec.expect_role}/{spec.expect_profile})")
        return g

    async def _acquire_extra(self, spec: SeatSpec) -> None:
        """Fill a seat's other slot. Best effort: a grant elsewhere (or none in 60s) is given back now."""
        reply = await self.transport.hold(spec.work_class, self.actor)
        detail = reply.get("detail") or {}
        lease_id = detail.get("lease_id")
        if not lease_id:
            return
        self._held.append(lease_id)
        self.ledger.append(action="acquired", lease_id=lease_id, model=spec.model_key,
                           work_class=spec.work_class, status=detail.get("status"), extra_slot=True)
        grant = detail.get("grant") if detail.get("status") == "granted" else None
        if grant is None and reply.get("ok"):
            grant = await self.transport.wait_granted(lease_id, 60.0)
        if not grant or grant.get("role") != spec.expect_role:
            await self._release_one(lease_id, reason="extra_slot_elsewhere")

    async def _release_one(self, lease_id: str, *, reason: str) -> None:
        ok = False
        for verb in ("release", "cancel"):
            try:
                res = await getattr(self.transport, verb)(lease_id, self.actor)
                ok = lease_ended(res)
            except Exception as exc:  # noqa: BLE001 -- keep going; the ledger keeps the id
                res = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            if ok:
                break
        self.ledger.append(action="released" if ok else "release_failed", lease_id=lease_id, reason=reason)
        if lease_id in self._held:
            self._held.remove(lease_id)

    async def release_all(self, *, reason: str) -> None:
        """Every hold goes back even if this task is cancelled (again) mid-release: each release is
        shielded, a CancelledError is held until the loop is done, then re-raised."""
        pending: Optional[BaseException] = None
        if self._unknown:
            self._unknown = False
            try:
                await asyncio.shield(self._sweep_unknown())
            except asyncio.CancelledError as exc:
                pending = exc
        for lease_id in list(reversed(self._held)):
            try:
                await asyncio.shield(self._release_one(lease_id, reason=reason))
            except asyncio.CancelledError as exc:
                pending = exc
        self.seats.clear()
        if pending is not None:
            raise pending


_ENDED_STATUSES = ("released", "cancelled", "dead_lettered", "expired", "aborted")


def lease_ended(reply: dict[str, Any]) -> bool:
    """Did this control reply leave the lease ended? `release` on a queued lease ends it as cancelled
    and answers ok=False/status=unavailable (runtime._reply_for); `cancel` on an already-ended lease
    answers `not_cancelable_from_<status>`. Both mean: nothing is held any more."""
    if reply.get("ok"):
        return True
    detail = reply.get("detail") or {}
    reason = str(reply.get("reason") or detail.get("reason") or "")
    if detail.get("status") in ("ok", "unknown_lease") or reason == "unknown_lease":
        return True
    if detail.get("status") == "unavailable" and any(s in reason for s in _ENDED_STATUSES + ("cancel",)):
        return True
    return any(reason == f"not_cancelable_from_{s}" for s in _ENDED_STATUSES)


HOLDER = "operator:bonsai-replay-eval"


def gateway_headers(model: str, grant: Grant, holder: str = HOLDER) -> dict[str, str]:
    """The headers every model call of this task carries to orion-llm-gateway: the task's hold ref.
    The gateway attaches the call as a child of the hold (pool verb ``attach``), exactly how a
    durable run's calls ride its hold, so the pool never lends the slot away mid-generation."""
    from orion.llm.resource_lease import GPU_LEASE_HEADER, encode_gpu_lease_header
    from orion.schemas.gpu_pool import GpuLeaseRefV1

    ref = GpuLeaseRefV1(lease_id=grant.lease_id, generation=grant.generation, role=grant.role, holder=holder)
    return {GPU_LEASE_HEADER: encode_gpu_lease_header(ref)}


def gateway_route(model: str) -> str:
    """The gateway route a model's calls name: its seat's operator class (config/gpu_pool.yaml)."""
    return SEATS[model].work_class


LIVE_STATUSES = ("granted", "recalling", "queued", "retry_wait", "backlogged")


def live_holder_leases(pool_state: dict[str, Any], holder: str = HOLDER) -> list[str]:
    """Lease ids the pool's own state (GET /v1/pool) shows live for ``holder``."""
    return sorted(str(lease["lease_id"]) for lease in pool_state.get("leases") or []
                  if lease.get("holder") == holder and lease.get("status") in LIVE_STATUSES)


def http_sweeper(pool_http: str, holder: str = HOLDER) -> Callable[[], Any]:
    async def sweep() -> list[str]:
        import httpx

        async with httpx.AsyncClient(timeout=15) as c:
            return live_holder_leases((await c.get(f"{pool_http}/v1/pool")).json(), holder)

    return sweep


async def release_leftovers(transport: ControlTransport, ledger: HoldLedger, actor: str,
                            sweep: Optional[Callable[[], Any]] = None) -> list[str]:
    """Release every lease the ledger says was taken and never released, plus (``sweep``) every live
    lease the pool itself shows for this holder -- covers a hold whose reply never arrived."""
    holds = PoolHolds(transport=transport, ledger=ledger, actor=actor, sweep=sweep)
    holds._held = ledger.outstanding()
    holds._unknown = sweep is not None
    if sweep is not None:
        await holds._sweep_unknown()
        holds._unknown = False
    left = list(holds._held)
    await holds.release_all(reason="leftover")
    return left


# run_subscription name of the grant listener; one BusControlTransport open per process at a time.
_POOL_EVENTS_LISTENER = "model-replay-pool-events"


class BusControlTransport:
    """The real transport: pool control verbs over the Orion bus (same envelope as gpu_pool_pause.py)."""

    def __init__(self, bus_url: str, timeout_sec: float = 15.0) -> None:
        self.bus_url = bus_url
        self.timeout_sec = timeout_sec
        self._bus: Any = None
        self._grants: dict[str, asyncio.Future] = {}
        self._listener: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "BusControlTransport":
        from orion.core.bus.async_service import OrionBusAsync

        self._bus = OrionBusAsync(url=self.bus_url)
        await self._bus.connect()
        opened_at = time.time()
        self._listener = asyncio.create_task(self._listen())
        try:
            await asyncio.wait_for(self._subscribed(opened_at), timeout=self.timeout_sec)
        except BaseException:
            await self.__aexit__()  # the listener retries forever; do not leave it behind
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._listener:
            self._listener.cancel()
            with suppress(asyncio.CancelledError):
                await self._listener
        await self._bus.close()

    @property
    def bus(self) -> Any:
        return self._bus

    async def _listen(self) -> None:
        from orion.core.bus.resilience import run_subscription
        from orion.schemas.gpu_pool import GPU_POOL_EVENT_CHANNEL

        # Reopens a dropped subscription; a grant published while it is down is missed and
        # that hold's wait_granted times out, as for any lost grant.
        await run_subscription(self._bus, GPU_POOL_EVENT_CHANNEL, self._on_pool_event,
                               stop_event=None, name=_POOL_EVENTS_LISTENER)

    @staticmethod
    async def _subscribed(since: float) -> None:
        """Holds go out only once the grant listener is subscribed, so no grant is missed.
        ``since`` skips a row left by an earlier transport in this process."""
        from orion.core.bus.resilience import subscription_health

        while not any(r["name"] == _POOL_EVENTS_LISTENER and r["subscribed"]
                      and (r["last_subscribed_at"] or 0) >= since
                      for r in subscription_health()["listeners"]):
            await asyncio.sleep(0.02)

    def _on_pool_event(self, msg: dict[str, Any]) -> None:
        try:
            payload = self._bus.codec.decode(msg["data"]).envelope.payload or {}
        except Exception:  # noqa: BLE001
            return
        lid = payload.get("lease_id") or ""
        if payload.get("event") == "granted" and (payload.get("detail") or {}).get("grant"):
            result: Optional[dict[str, Any]] = payload["detail"]["grant"]
        elif payload.get("event") in ("unavailable", "backlogged", "cancelled", "dead_lettered",
                                      "expired", "aborted"):
            result = None
        else:
            return
        # A grant can be published before our hold RPC's reply is read: keep it either way.
        fut = self._grants.setdefault(lid, asyncio.get_running_loop().create_future())
        if not fut.done():
            fut.set_result(result)

    async def _control(self, **fields: Any) -> dict[str, Any]:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
        from orion.schemas.gpu_pool import (
            GPU_POOL_CONTROL_KIND, GPU_POOL_CONTROL_REPLY_PREFIX, GPU_POOL_CONTROL_REQUEST_CHANNEL,
            GpuPoolControlReplyV1, GpuPoolControlV1,
        )

        reply_channel = f"{GPU_POOL_CONTROL_REPLY_PREFIX}{uuid.uuid4().hex}"
        ctl = GpuPoolControlV1(**fields)
        env = BaseEnvelope(kind=GPU_POOL_CONTROL_KIND, source=ServiceRef(name="operator:bonsai-replay-eval"),
                           correlation_id=uuid.uuid4(), reply_to=reply_channel, payload=ctl.model_dump(mode="json"))
        raw = await self._bus.rpc_request(GPU_POOL_CONTROL_REQUEST_CHANNEL, env, reply_channel=reply_channel,
                                          timeout_sec=self.timeout_sec)
        data = raw.get("data") if isinstance(raw, dict) and "data" in raw else raw
        decoded = json.loads(data) if isinstance(data, (str, bytes, bytearray)) else data
        payload = decoded.get("payload") if isinstance(decoded, dict) else None
        return GpuPoolControlReplyV1.model_validate(payload).model_dump(mode="json")

    async def hold(self, work_class: str, actor: str) -> dict[str, Any]:
        res = await self._control(verb="hold", work_class=work_class, actor=actor)
        lid = (res.get("detail") or {}).get("lease_id")
        if lid and lid not in self._grants:
            self._grants[lid] = asyncio.get_running_loop().create_future()
        return res

    async def release(self, lease_id: str, actor: str) -> dict[str, Any]:
        return await self._control(verb="release", lease_id=lease_id, actor=actor)

    async def cancel(self, lease_id: str, actor: str) -> dict[str, Any]:
        return await self._control(verb="cancel", lease_id=lease_id, actor=actor)

    async def wait_granted(self, lease_id: str, timeout_sec: float) -> Optional[dict[str, Any]]:
        fut = self._grants.setdefault(lease_id, asyncio.get_running_loop().create_future())
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout_sec)
        except asyncio.TimeoutError:
            return None


class LeaseHoldTransport(BusControlTransport):
    """The replay's transport (leased evals, 2026-11-12): holds through the lease RPC at ``background``,
    non-retryable, heartbeated, on the pinned operator classes. ``recalled`` is set once the pool asks
    for any held seat back (or loses it); the runner then stops the run and writes a partial report.
    Grants still arrive on the pool event channel (``wait_granted``, inherited)."""

    def __init__(self, bus_url: str, timeout_sec: float = 15.0, *, holder: str = HOLDER,
                 heartbeat_sec: float = 30.0, queue_deadline_sec: float = 3 * 3600.0) -> None:
        super().__init__(bus_url, timeout_sec)
        self.holder, self.heartbeat_sec, self.queue_deadline_sec = holder, heartbeat_sec, queue_deadline_sec
        self.recalled = asyncio.Event()
        self.recall_reason: Optional[str] = None
        self._beats: dict[str, asyncio.Task] = {}

    async def __aexit__(self, *exc: Any) -> None:
        for task in list(self._beats.values()):
            task.cancel()
        self._beats.clear()
        await super().__aexit__(*exc)

    async def _rpc(self, req: Any, ok: tuple[str, ...] = ("ok",)) -> dict[str, Any]:
        """A lease-RPC reply in the control-reply shape PoolHolds reads (``ok``, ``reason``, ``detail``)."""
        from orion.gpu_pool.client import lease_rpc

        reply = await lease_rpc(self._bus, req, source=self.holder, timeout_sec=self.timeout_sec)
        return {"ok": reply.status in ok, "reason": reply.reason, "detail": reply.model_dump(mode="json")}

    def mark_recalled(self, loop: asyncio.AbstractEventLoop, why: str) -> None:
        """Thread-safe: a call found the hold gone before the heartbeat did (the gateway refused it)."""
        def _set() -> None:
            self.recall_reason = self.recall_reason or f"hold_gone:{why[:120]}"
            self.recalled.set()
        loop.call_soon_threadsafe(_set)

    def _start_beat(self, lease_id: str) -> None:
        if lease_id not in self._beats:
            self._beats[lease_id] = asyncio.create_task(self._beat(lease_id))

    def _stop_beat(self, lease_id: str) -> None:
        task = self._beats.pop(lease_id, None)
        if task is not None:
            task.cancel()

    async def _beat(self, lease_id: str) -> None:
        from orion.gpu_pool.client import heartbeat_lease

        while True:
            await asyncio.sleep(self.heartbeat_sec)
            try:
                reply = await heartbeat_lease(self._bus, lease_id, source=self.holder, timeout_sec=self.timeout_sec)
            except Exception:  # noqa: BLE001 -- a missed beat is survivable; the TTL is 3x this
                continue
            if reply.status != "granted":
                self.recall_reason = f"{lease_id}:{reply.status}" + (f":{reply.reason}" if reply.reason else "")
                self.recalled.set()
                if reply.status != "recall":
                    return

    async def hold(self, work_class: str, actor: str) -> dict[str, Any]:
        from datetime import datetime, timedelta, timezone

        from orion.schemas.gpu_pool import GpuLeaseRequestV1

        req = GpuLeaseRequestV1(
            verb="acquire", request_id=f"{self.holder}:{uuid.uuid4().hex}", holder=self.holder,
            work_class=work_class, priority="background", kind="hold", retryable=False,
            deadline_at=datetime.now(timezone.utc) + timedelta(seconds=self.queue_deadline_sec))
        res = await self._rpc(req, ok=("granted", "queued"))
        detail = res["detail"]
        lid = detail.get("lease_id")
        if lid and lid not in self._grants:
            self._grants[lid] = asyncio.get_running_loop().create_future()
        if lid and detail.get("status") == "granted":
            self._start_beat(lid)
        return res

    async def wait_granted(self, lease_id: str, timeout_sec: float) -> Optional[dict[str, Any]]:
        grant = await super().wait_granted(lease_id, timeout_sec)
        if grant is not None:
            self._start_beat(lease_id)
        return grant

    async def release(self, lease_id: str, actor: str) -> dict[str, Any]:
        """Release one hold. Once any seat of this run was recalled, every release says
        ``cancelled`` (the run stopped because of the recall), not ``ok``."""
        from orion.schemas.gpu_pool import GpuLeaseRequestV1

        self._stop_beat(lease_id)
        outcome = "cancelled" if self.recalled.is_set() else "ok"
        return await self._rpc(GpuLeaseRequestV1(verb="release", lease_id=lease_id, outcome=outcome))

    async def cancel(self, lease_id: str, actor: str) -> dict[str, Any]:
        from orion.schemas.gpu_pool import GpuLeaseRequestV1

        self._stop_beat(lease_id)
        return await self._rpc(GpuLeaseRequestV1(verb="cancel", lease_id=lease_id))


def seat_problem(pool_state: dict[str, Any], models: tuple[str, ...] = ACQUIRE_ORDER, holder: str = HOLDER) -> Optional[str]:
    """Why the replay should not take holds right now (None = go): a seat is not serving the expected
    model, or another run already holds it (one hold per role in stage 7.2, so our hold would queue
    on gpu1 or fall through to hecate/chat for gpu2)."""
    roles = {r.get("role"): r for r in pool_state.get("roles") or []}
    for m in models:
        spec = SEATS[m]
        r = roles.get(spec.expect_role) or {}
        if r.get("status") != "confirmed" or r.get("profile_name") != spec.expect_profile:
            return f"{spec.expect_role} is {r.get('status')}/{r.get('profile_name')}, want {spec.expect_profile}"
        busy = [lease.get("holder") for lease in pool_state.get("leases") or []
                if lease.get("role") == spec.expect_role and lease.get("kind") == "hold"
                and lease.get("status") in ("granted", "recalling") and lease.get("holder") != holder]
        if busy:
            return f"{spec.expect_role} is held by {busy[0]}"
    if pool_state.get("actuation_paused"):
        return "pool actuation is paused"
    return None


def http_seat_check(pool_http: str) -> Callable[[], Any]:
    async def check() -> Optional[str]:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=15) as c:
                return seat_problem((await c.get(f"{pool_http}/v1/pool")).json())
        except Exception as exc:  # noqa: BLE001
            return f"pool state unreadable: {type(exc).__name__}"

    return check
