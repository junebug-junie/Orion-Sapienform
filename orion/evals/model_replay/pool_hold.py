"""Operator holds that pin each model's card for one replay task, released on every exit path.

WHY OPERATOR HOLDS. A caller cannot ask the pool for a role or a profile, only a work class
(`config/gpu_pool.yaml` classes). So the replay pins cards by class and then CHECKS what it got:

  * Q4 (gpu1):     class ``memory_distill`` -> roles [agent] only. agent has 1 slot, so one hold
                   makes gpu1 ours alone. Verified: grant.role == "agent" and grant.profile_name is
                   the Q4 profile.
  * Bonsai (gpu2): class ``agent`` -> [agent, agent-gpu2, agent-deep, chat]. Taken AFTER the gpu1
                   hold, so agent is full and the class falls through to agent-gpu2. Verified:
                   grant.role == "agent-gpu2" and grant.profile_name is the Bonsai profile. Only one
                   hold: the pool allows one non-urgent hold per role (stage 7.2), so a second one
                   would land on hecate or chat. gpu2's other slot can serve one-off calls, as in
                   production; the replay's own calls ride the hold as attached child leases.

A grant on the wrong role or profile is released immediately and the task is retried later
(runner: the seat check waits until gpu1 and gpu2 carry no other run's hold). Because class
``agent`` CAN fall through to agent-deep (hecate) or chat when gpu2 is busy, such a grant may
exist for the length of one RPC round trip before it is given back. An operator-only class
covering just agent-gpu2 would remove that; it is a config/gpu_pool.yaml change, not made here.

Operator holds have no heartbeat and no expiry (``lease_graph.py``: operator -> expiry None), so a
leaked hold stays until the role's ``max_hold_sec`` or forever on gpu1. Hence:
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


SEATS: dict[str, SeatSpec] = {
    "q4": SeatSpec("q4", "memory_distill", "agent", Q4_PROFILE),
    "bonsai": SeatSpec("bonsai", "agent", "agent-gpu2", BONSAI_PROFILE),
}
# gpu1 first: with agent's only slot held, class `agent` cannot land on agent.
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
                spec = SEATS[key]
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


def attach_factory(bus: Any, model: str, grant: Grant) -> Callable[[float], Any]:
    """Each model call as a child lease of the task's hold (pool verb ``attach``), exactly how a
    durable run's calls ride its hold. Without it the pool sees an idle hold and lends the slot to
    one-off calls in the middle of a replayed generation (gap sharing), slowing one model at random."""
    from orion.gpu_pool.client import gpu_lease
    from orion.schemas.gpu_pool import GpuLeaseRefV1

    ref = GpuLeaseRefV1(lease_id=grant.lease_id, generation=grant.generation, role=grant.role, holder=HOLDER)

    def enter(timeout_sec: float) -> Any:
        return gpu_lease(bus, work_class=SEATS[model].work_class, holder=HOLDER, priority="interactive",
                         kind="request", deadline_sec=max(30.0, min(600.0, timeout_sec)), hold=ref)

    return enter


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


class BusControlTransport:
    """The real transport: pool control verbs over the Orion bus (same envelope as gpu_pool_pause.py)."""

    def __init__(self, bus_url: str, timeout_sec: float = 15.0) -> None:
        self.bus_url = bus_url
        self.timeout_sec = timeout_sec
        self._bus: Any = None
        self._grants: dict[str, asyncio.Future] = {}
        self._listener: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()

    async def __aenter__(self) -> "BusControlTransport":
        from orion.core.bus.async_service import OrionBusAsync

        self._bus = OrionBusAsync(url=self.bus_url)
        await self._bus.connect()
        self._listener = asyncio.create_task(self._listen())
        await asyncio.wait_for(self._ready.wait(), timeout=self.timeout_sec)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._listener:
            self._listener.cancel()
        await self._bus.close()

    @property
    def bus(self) -> Any:
        return self._bus

    async def _listen(self) -> None:
        from orion.schemas.gpu_pool import GPU_POOL_EVENT_CHANNEL

        async with self._bus.subscribe(GPU_POOL_EVENT_CHANNEL) as pubsub:
            self._ready.set()
            async for msg in self._bus.iter_messages(pubsub):
                try:
                    payload = self._bus.codec.decode(msg["data"]).envelope.payload or {}
                except Exception:  # noqa: BLE001
                    continue
                lid = payload.get("lease_id") or ""
                if payload.get("event") == "granted" and (payload.get("detail") or {}).get("grant"):
                    result: Optional[dict[str, Any]] = payload["detail"]["grant"]
                elif payload.get("event") in ("unavailable", "backlogged", "cancelled", "dead_lettered",
                                              "expired", "aborted"):
                    result = None
                else:
                    continue
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
