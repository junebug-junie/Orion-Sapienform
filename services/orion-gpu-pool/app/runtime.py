"""The pool runtime: the single writer that joins discovery, the scheduler, the lease graph,
the fenced projection and the bus.

Every entrypoint (RPC verb, announcement, tick) runs under one asyncio lock, so decisions are
serial and the scheduler always sees a consistent world. Only the Postgres advisory-lock holder
runs a runtime at all (see app/main.py), so this lock is the whole concurrency story.

Stage 4.3 adds durable-run holds (acquire kind=hold, ``attach`` for each call under one, ``status``
for resume) and the swap actuation engine: for a seat named in GPU_POOL_ACTUATE_ROLES the pool
sends GpuActuateV1 to the seat's host actuator and walks the card through
idle -> loading|unloading -> idle|fault from the GpuActuateResultV1 replies, persisting every step
on gpu_pool_cards so a restart reconciles (``status``) instead of issuing a second transition.
Seats not in that set keep today's observe behaviour: ``swap_requested {actuated: false}``.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.gpu_pool.client import WAIT_HOP_LABEL
from orion.gpu_pool.config import PoolConfig, launch_digest
from orion.gpu_pool.discovery import Probe, resolve_roles
from orion.gpu_pool.lease_graph import FINAL, InvalidTransition, initial_state
from orion.gpu_pool.scheduler import (
    Abort, Backlog, CardLive, DeadLetter, Expire, Grant, LeaseView, Recall, Requeue, RoleLive,
    SwapBlocked, SwapLoad, SwapUnload, Unavailable, schedule,
)
from orion.gpu_pool.config import SWAP_GUARDS
from orion.schemas.gpu_pool import (
    GPU_ACTUATE_KIND, GPU_POOL_ACTUATE_REQUEST_CHANNEL, GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND,
    GPU_POOL_STATE_CHANNEL, GPU_POOL_STATE_KIND,
    DiscoveredRoleV1, GpuActuateResultV1, GpuActuateV1, GpuCardStateV1, GpuLeaseGrantV1, GpuLeaseReplyV1,
    GpuLeaseRequestV1, GpuLeaseRowV1, GpuPoolControlReplyV1, GpuPoolControlV1, GpuPoolEventV1, GpuPoolStateV1,
    LlmWorkerAnnounceV1,
)

logger = logging.getLogger("orion-gpu-pool.runtime")

Prober = Callable[[str, str, str, str], Awaitable[Probe]]

# scheduler decision -> lease-graph event type
_EVENT_FOR = {Grant: "grant", Recall: "recall", Abort: "abort", Expire: "expire",
              Unavailable: "unavailable", Backlog: "backlog", Requeue: "requeue", DeadLetter: "dead_letter"}
# lease-graph status reached -> public event name
_PUBLIC = {"granted": "granted", "recalling": "recalled", "backlogged": "backlogged",
           "unavailable": "unavailable", "dead_letter": "dead_lettered", "released": "released",
           "retry_wait": "retried", "queued": "queued"}
# Grammar carries the EXCEPTIONS only. Routine grants would add one grammar row per LLM call once
# the gateway leases (stage 3) to a ~475k/day, 3-day-retention table, for a fact gpu_pool_events
# already holds. Every event still goes to gpu_pool_events via orion:gpu_pool:event.
_GRAMMAR_EVENTS = {"recalled", "aborted", "expired", "backlogged", "unavailable",
                   "dead_lettered", "swap_requested", "discovery_mismatch", "swap_failed", "actuate_refused"}
# While a `status` reply says the action is still running, the pool asks again this often. A
# missed deadline is not a failure then: the actuator's own worst case (drain + ready wait +
# docker) can exceed the role's launch.timeout_sec, and its transition has budgets of its own.
STATUS_POLL_SEC = 30.0
# How long a `status` may take to answer before the actuator counts as unreachable and the card
# faults. Not actuate_ack_sec: the circe actuator answers status only after `docker` has reported
# the containers (up to ~60s, services/orion-gpu-lane-controller/app/actuator_bus.py).
STATUS_REPLY_SEC = 90.0
# A status reply that cannot say whether the action is running (an actuator predating the
# `in_flight` field) and shows a half-done card is re-asked this many times before the card faults.
MAX_STATUS_EXTENSIONS = 4


def validate_actuate_roles(cfg: PoolConfig, roles: Iterable[str]) -> frozenset[str]:
    """GPU_POOL_ACTUATE_ROLES must name non-operator swap seats with a launch block (the launch
    names the actuator). A typo must fail the boot, not silently leave a seat unactuated."""
    out = frozenset(r.strip() for r in roles if r and r.strip())
    problems = []
    for role in sorted(out):
        spec = cfg.roles.get(role)
        if spec is None:
            problems.append(f"{role}: not a role in config/gpu_pool.yaml")
        elif spec.swap is None or spec.operator_only:
            problems.append(f"{role}: not a (non-operator) swap seat")
        elif spec.launch is None:
            problems.append(f"{role}: has no launch block naming its actuator")
    if problems:
        raise ValueError("GPU_POOL_ACTUATE_ROLES: " + "; ".join(problems))
    return out


def _ts(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _action(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


SLOW_LOCK_MS = 250.0


class LockStats:
    """Per-op count / worst wait / worst hold since the last read -- /health reports and resets it,
    so a stall shows up without grepping logs."""

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, float]] = {}

    def observe(self, op: str, waited_ms: float, held_ms: float) -> None:
        st = self._stats.setdefault(op, {"n": 0, "max_wait_ms": 0.0, "max_hold_ms": 0.0, "slow": 0})
        st["n"] += 1
        st["max_wait_ms"] = max(st["max_wait_ms"], round(waited_ms, 1))
        st["max_hold_ms"] = max(st["max_hold_ms"], round(held_ms, 1))
        if waited_ms >= SLOW_LOCK_MS or held_ms >= SLOW_LOCK_MS:
            st["slow"] += 1

    def drain(self) -> dict[str, dict[str, float]]:
        out, self._stats = self._stats, {}
        return out


class PoolRuntime:
    def __init__(self, *, cfg: PoolConfig, profiles: dict[str, Any], store: Any, graph: Any,
                 bus: Any = None, prober: Prober | None = None, now: Callable[[], datetime] = _utcnow,
                 mode: str = "observe", service_name: str = "orion-gpu-pool",
                 announce_stale_sec: float = 120.0, probe_interval_sec: float = 15.0,
                 state_publish_sec: float = 5.0, replay_payload_max_bytes: int = 262144,
                 actuate_roles: Iterable[str] = ()):

        self.cfg, self.profiles, self.store, self.graph, self.bus = cfg, profiles, store, graph, bus
        self.prober, self.now, self.mode = prober, now, mode
        self.service_name = service_name
        self.announce_stale_sec, self.probe_interval_sec = announce_stale_sec, probe_interval_sec
        self.state_publish_sec, self.replay_payload_max_bytes = state_publish_sec, replay_payload_max_bytes
        self.lock = asyncio.Lock()
        self._lock_holder: str | None = None
        self._phases: dict[str, float] = {}
        self.lock_stats = LockStats()
        self.cards: dict[str, CardLive] = {}
        self.announcements: dict[str, LlmWorkerAnnounceV1] = {}
        self.probes: dict[str, Probe] = {}
        self.discovered: list[DiscoveredRoleV1] = []
        self.roles: dict[str, RoleLive] = {}
        self.unclaimed: list[str] = []
        self._last_probe: datetime | None = None
        self._last_state: datetime | None = None
        self._swap_requested: set[tuple] = set()
        self._ctx_seen: dict[str, int] = {}
        self.actuate_roles = validate_actuate_roles(cfg, actuate_roles)
        # Swap-load guards (app/guards.py fills these outside the lock). Every guard starts failing
        # ("unread"): a pool that has not read the thermal sensor yet must not load a model.
        self.guard_states: dict[str, str | None] = {g: "unread" for g in SWAP_GUARDS}
        self._started = False

    # --- lifecycle --------------------------------------------------------------------
    async def start(self) -> None:
        stored = {r["card"]: r for r in await self.store.cards()}
        for card in self.cfg.cards:
            row = stored.get(card)
            if row is None:
                await self.store.upsert_card({"card": card, "lent": False, "swapped_in": [],
                                              "swap_state": "idle", "updated_at": self.now(),
                                              "updated_by": "boot"})
                self.cards[card] = CardLive(card)
            else:
                self.cards[card] = CardLive(
                    card, bool(row["lent"]), set(row["swapped_in"] or []), row["swap_state"],
                    _ts(row.get("cooldown_until")), _ts(row.get("last_active_at")),
                    swap_role=row.get("swap_role"), residency_until=_ts(row.get("residency_until")),
                    loaded_at=_ts(row.get("loaded_at")), swap_generation=int(row.get("swap_generation") or 0),
                    swap_action=_action(row.get("swap_action")))
        for row in await self.store.live_leases():
            await self._sync_row(row["lease_id"])
        self._resolve()
        self._started = True
        # A swap in flight when the previous process died: ask the actuator what happened rather
        # than issue a second transition (acceptance check 6).
        for seat, act in self._pending_actions().items():
            logger.warning("gpu_pool_actuation_reconcile seat=%s action_id=%s state=%s",
                           seat, act.get("action_id"), self.cards[self.cfg.roles[seat].cards[0]].swap_state)
            await self._send_status(seat, act, reason="pool_restart")

    def _thread(self, lease_id: str) -> dict:
        return {"configurable": {"thread_id": f"gpu_pool:{lease_id}"}}

    # --- discovery --------------------------------------------------------------------
    @contextlib.asynccontextmanager
    async def _locked(self, op: str):
        """The runtime lock, timed. Every lease verb and the tick serialise on it, so one slow
        holder stalls every caller (live 2026-09-25: bursts of ~2 s waits per acquire). A wait or
        hold over SLOW_LOCK_MS is logged with what held it -- the evidence, not a guess."""
        t0 = time.monotonic()
        async with self.lock:
            waited = (time.monotonic() - t0) * 1000
            t1 = time.monotonic()
            prev, self._lock_holder = self._lock_holder, op
            self._phases = {}
            try:
                yield
            finally:
                self._lock_holder = prev
                held = (time.monotonic() - t1) * 1000
                self.lock_stats.observe(op, waited, held)
                if waited >= SLOW_LOCK_MS or held >= SLOW_LOCK_MS:
                    logger.warning("gpu_pool_slow_lock op=%s waited_ms=%.0f held_ms=%.0f phases=%s",
                                   op, waited, held, self._phases or "-")

    def _phase(self, name: str, started: float) -> None:
        self._phases[name] = round(self._phases.get(name, 0.0) + (time.monotonic() - started) * 1000, 1)

    async def on_announce(self, ann: LlmWorkerAnnounceV1) -> None:
        async with self._locked("on_announce"):
            self.announcements[ann.role] = ann

    async def _probe_all(self) -> None:
        if self.prober is None:
            return
        for role, spec in self.cfg.roles.items():
            url = self.cfg.url(role)
            try:
                self.probes[role] = await self.prober(role, url, spec.kind, spec.health)
            except Exception as exc:  # noqa: BLE001
                self.probes[role] = Probe(False, error=f"{type(exc).__name__}: {exc}", checked_at=self.now())
        self._last_probe = self.now()

    def _seat_alive(self, role: str) -> bool:
        """Its worker is really up: announced fresh AND answering /props."""
        ann = self.announcements.get(role)
        probe = self.probes.get(role)
        return bool(ann and (self.now() - ann.announced_at).total_seconds() <= self.announce_stale_sec
                    and probe and probe.ok and probe.props)

    def _pool_owns(self, seat: str) -> bool:
        """The pool's own swap state is the truth for a seat it actuates, from the first action it
        issued there (or whenever that card is mid-swap / faulted)."""
        if seat not in self.actuate_roles:
            return False
        cards = [self.cards[c] for c in self.cfg.roles[seat].cards]
        return any(c.swap_generation > 0 or c.swap_state != "idle" for c in cards)

    def _observe_swap_seats(self) -> None:
        """A seat the pool does not actuate is loaded when its worker is really up. Otherwise the
        pool would refuse to use a 27B that the old elastic runtime (or an operator) already brought
        up on gpu2. A seat the pool actuates is observed only until the pool's first action there:
        that is how a seat loaded by the old path is adopted when actuation is armed, not reloaded
        (stage 4 spec, "No window where nobody can open gpu2"). After that the pool's intent is the
        truth, and discovery still gates grants on the worker being confirmed."""
        if self.mode == "enforce":
            return
        for role, spec in self.cfg.roles.items():
            if spec.swap is None or self._pool_owns(role):
                continue
            alive = self._seat_alive(role)
            for card in spec.cards:
                c = self.cards[card]
                if alive:
                    if role in self.actuate_roles and role not in c.swapped_in:
                        # Adopted: its max_hold_sec and idle-unload clocks start now. Without the
                        # second, an armed pool would unload a 27B the old path just loaded on
                        # its first tick, before the run that asked for it could be granted.
                        c.loaded_at = self.now()
                        c.last_active_at = self.now()
                    c.swapped_in.add(role)
                else:
                    if role in c.swapped_in and role in self.actuate_roles:
                        c.loaded_at = None
                    c.swapped_in.discard(role)

    def _resolve(self) -> None:
        self._observe_swap_seats()
        before = {d.role: d.status for d in self.discovered}
        self.discovered, self.roles, self.unclaimed = resolve_roles(
            self.cfg, self.profiles, self.announcements, self.probes, self.cards, self.now(),
            self.announce_stale_sec)
        # Remember each role's last-seen context size. A role that is restarting reports none;
        # forgetting it would make its class look smaller and wrongly refuse big prompts
        # ("min_ctx_exceeds_class") while the one role that fits them is briefly away.
        for name, live in self.roles.items():
            if live.ctx_per_slot:
                self._ctx_seen[name] = live.ctx_per_slot
        self._discovery_changes = [
            d for d in self.discovered
            if d.status in ("confirmed", "mismatch") and before.get(d.role) != d.status
        ]

    # --- RPC verbs --------------------------------------------------------------------
    async def acquire(self, req: GpuLeaseRequestV1, *, operator: bool = False) -> GpuLeaseReplyV1:
        async with self._locked("acquire"):
            if not req.work_class or req.work_class not in self.cfg.classes:
                return GpuLeaseReplyV1(status="unavailable", reason=f"unknown_class:{req.work_class}")
            roles = self.cfg.classes[req.work_class].roles
            if any(self.cfg.roles[r].operator_only for r in roles) and not operator:
                return GpuLeaseReplyV1(status="unavailable", reason="operator_only_class")
            request_id = req.request_id or uuid.uuid4().hex
            existing = await self.store.lease_by_request(request_id)
            if existing is not None:
                return await self._reply_for(existing)
            if req.replay_payload is not None and \
                    len(json.dumps(req.replay_payload)) > self.replay_payload_max_bytes:
                return GpuLeaseReplyV1(status="unavailable", reason="replay_payload_too_large")
            lease_id = uuid.uuid4().hex
            now = self.now()
            request = {**req.model_dump(mode="json", exclude={"verb", "lease_id", "outcome", "detail"}),
                       "request_id": request_id, "holder": req.holder or "unknown", "operator": operator}
            t = time.monotonic()
            await self._start_thread(lease_id, request, now)
            self._phase("start_thread", t)
            await self._schedule_and_apply()
            return await self._reply_for(await self.store.lease(lease_id))

    async def attach(self, req: GpuLeaseRequestV1) -> GpuLeaseReplyV1:
        """A call made under a durable run's hold: a request lease that runs in the hold's slot
        (H2), idempotent on request_id. The hold must be live at the generation the caller holds:
        a re-granted hold means the caller's view is stale and it must not attach."""
        async with self._locked("attach"):
            existing = await self.store.lease_by_request(req.request_id)
            if existing is not None:
                return await self._reply_for(existing)
            if not req.work_class or req.work_class not in self.cfg.classes:
                return GpuLeaseReplyV1(status="unavailable", reason=f"unknown_class:{req.work_class}")
            # Refusals name the hold only in `reason`, NEVER in `lease_id`: a reply's lease_id is the
            # caller's own lease, and the client cancels that on failure -- echoing the hold's id
            # there would let one stale call cancel its whole run's hold.
            hold = await self.store.lease(req.hold_lease_id)
            if hold is None:
                return GpuLeaseReplyV1(status="unknown_lease", reason="hold_unknown")
            if hold.get("kind") != "hold":
                return GpuLeaseReplyV1(status="unavailable", reason="not_a_hold")
            if hold["status"] not in ("granted", "recalling"):
                return GpuLeaseReplyV1(status="unavailable", reason=f"hold_not_granted:{hold['status']}")
            if int(hold.get("generation") or 0) != req.hold_generation:
                return GpuLeaseReplyV1(status="unavailable",
                                       reason=f"stale_hold_generation:{hold.get('generation')}")
            if req.replay_payload is not None and \
                    len(json.dumps(req.replay_payload)) > self.replay_payload_max_bytes:
                return GpuLeaseReplyV1(status="unavailable", reason="replay_payload_too_large")
            lease_id = uuid.uuid4().hex
            request = {**req.model_dump(mode="json", exclude={"verb", "lease_id", "outcome", "detail"}),
                       "kind": "request", "holder": req.holder or hold["holder"], "operator": False}
            t = time.monotonic()
            await self._start_thread(lease_id, request, self.now())
            self._phase("start_thread", t)
            await self._schedule_and_apply()
            return await self._reply_for(await self.store.lease(lease_id))

    async def status(self, lease_id: str) -> GpuLeaseReplyV1:
        """Read-only: where ``lease_id`` stands (resume after restart, Door-A validation). No lock:
        the projection has one writer and this changes nothing."""
        row = await self.store.lease(lease_id)
        if row is None:
            return GpuLeaseReplyV1(status="unknown_lease", lease_id=lease_id)
        return await self._reply_for(row)

    async def heartbeat(self, lease_id: str) -> GpuLeaseReplyV1:
        async with self._locked("heartbeat"):
            row = await self.store.lease(lease_id)
            if row is None:
                return GpuLeaseReplyV1(status="unknown_lease", lease_id=lease_id)
            if row["status"] in ("granted", "recalling"):
                row = await self._resume(lease_id, {"type": "heartbeat"}) or row
            return await self._reply_for(row)

    async def release(self, lease_id: str, outcome: str = "ok", detail: str | None = None) -> GpuLeaseReplyV1:
        async with self._locked("release"):
            row = await self.store.lease(lease_id)
            if row is None:
                return GpuLeaseReplyV1(status="unknown_lease", lease_id=lease_id)
            if row["status"] in ("released",):
                return await self._reply_for(row)
            if row["status"] in ("granted", "recalling"):
                kind = "release_ok" if outcome == "ok" else "cancel" if outcome == "cancelled" else "release_failed"
            else:
                # The caller is done with it (finished after an abort, or gave up while it waited):
                # end it, so it is never granted again to nobody.
                kind = "release_ok" if outcome == "ok" else "cancel"
            row = await self._resume(lease_id, {"type": kind, "reason": detail or outcome}) or row
            await self._schedule_and_apply()
            return await self._reply_for(row)

    async def cancel(self, lease_id: str) -> GpuLeaseReplyV1:
        return await self.release(lease_id, "cancelled", "cancelled")

    async def control(self, ctl: GpuPoolControlV1) -> GpuPoolControlReplyV1:
        logger.info("gpu_pool_control verb=%s actor=%s card=%s lease=%s class=%s",
                    ctl.verb, ctl.actor, ctl.card, ctl.lease_id, ctl.work_class)
        if ctl.verb in ("lend", "unlend"):
            async with self._locked("control"):
                card = self.cards.get(ctl.card or "")
                if card is None or not self.cfg.cards[card.card].lendable:
                    return GpuPoolControlReplyV1(ok=False, reason="card_not_lendable")
                card.lent = ctl.verb == "lend"
                await self.store.upsert_card({"card": card.card, "lent": card.lent,
                                              "updated_at": self.now(), "updated_by": ctl.actor})
                await self._emit(GpuPoolEventV1(event="lent" if card.lent else "unlent",
                                                cards=[card.card], holder=ctl.actor))
                await self._schedule_and_apply()
            return GpuPoolControlReplyV1(ok=True, detail={"card": card.card, "lent": card.lent})
        if ctl.verb in ("replay", "cancel"):
            async with self._locked("control"):
                row = await self.store.lease(ctl.lease_id or "")
                if row is None:
                    return GpuPoolControlReplyV1(ok=False, reason="unknown_lease")
                event = {"type": "replay" if ctl.verb == "replay" else "cancel", "reason": f"operator:{ctl.actor}"}
                new = await self._resume(row["lease_id"], event)
                if new is None:
                    return GpuPoolControlReplyV1(ok=False, reason=f"not_{ctl.verb}able_from_{row['status']}")
                if ctl.verb == "replay":
                    await self._emit_row("replayed", new)
                await self._schedule_and_apply()
            return GpuPoolControlReplyV1(ok=True, detail={"lease_id": row["lease_id"], "status": new["status"]})
        if ctl.verb == "hold" and self.mode != "enforce":
            # Until the pool actuates swaps (stage 5), a hold would only drain every card it spans
            # and never load anything: a full LLM outage until someone clicks release.
            return GpuPoolControlReplyV1(ok=False, reason="hold_requires_swap_actuation")
        if ctl.verb == "hold":
            # Operator holds (e.g. the multi-card experiment seat): no heartbeat, bounded by the
            # role's max_hold_sec, released with verb=release.
            reply = await self.acquire(GpuLeaseRequestV1(
                verb="acquire", work_class=ctl.work_class, holder=f"operator:{ctl.actor}",
                priority="interactive", kind="hold"), operator=True)
            return GpuPoolControlReplyV1(ok=reply.status in ("granted", "queued"), reason=reply.reason,
                                         detail=reply.model_dump(mode="json"))
        if ctl.verb == "release":
            reply = await self.release(ctl.lease_id or "", "ok", f"operator:{ctl.actor}")
            return GpuPoolControlReplyV1(ok=reply.status == "ok", reason=reply.reason,
                                         detail=reply.model_dump(mode="json"))
        if ctl.verb == "backfill":
            return await self._backfill(ctl)
        return GpuPoolControlReplyV1(ok=False, reason="unknown_verb")

    async def _backfill(self, ctl: GpuPoolControlV1) -> GpuPoolControlReplyV1:
        """Replay a filtered set of past leases as linked child leases. ``preview`` counts only."""
        spec = ctl.backfill or {}
        limit = min(int(spec.get("limit", 100)), 1000)
        async with self._locked("_backfill"):
            rows = await self.store.find_leases(
                work_class=spec.get("work_class"), holder=spec.get("holder"), status=spec.get("status"),
                since=_ts(spec.get("since")), until=_ts(spec.get("until")), limit=limit)
            if spec.get("preview", True):
                return GpuPoolControlReplyV1(ok=True, detail={"would_replay": len(rows)})
            children, skipped = [], []
            for parent in rows:
                snap = await self.graph.aget_state(self._thread(parent["lease_id"]))
                request = dict((snap.values or {}).get("request") or {})
                if not request:
                    # history pruned (GPU_POOL_LEASE_RETENTION_HOURS): say so, don't drop it
                    skipped.append(parent["lease_id"])
                    continue
                request.update(request_id=uuid.uuid4().hex, parent_lease_id=parent["lease_id"],
                               deadline_at=None)
                child = uuid.uuid4().hex
                await self._start_thread(child, request, self.now(), parent_lease_id=parent["lease_id"])
                children.append(child)
            await self._schedule_and_apply()
        return GpuPoolControlReplyV1(ok=True, detail={"replayed": len(children), "children": children,
                                                       "skipped_no_history": skipped})

    # --- the tick ---------------------------------------------------------------------
    async def tick(self) -> None:
        async with self._locked("tick"):
            now = self.now()
            if self._last_probe is None or (now - self._last_probe).total_seconds() >= self.probe_interval_sec:
                t = time.monotonic()
                await self._probe_all()
                self._phase("probe", t)
            t = time.monotonic()
            self._resolve()
            self._phase("resolve", t)
            for d in self._discovery_changes:
                await self._emit(GpuPoolEventV1(
                    event="discovery_confirmed" if d.status == "confirmed" else "discovery_mismatch",
                    role=d.role, cards=d.cards, reason=d.detail,
                    detail={"profile_name": d.profile_name, "model_file": d.model_file}))
            t = time.monotonic()
            await self._check_actuation()
            await self._clear_faults()
            self._phase("actuation", t)
            await self._schedule_and_apply()
            if self._last_state is None or (now - self._last_state).total_seconds() >= self.state_publish_sec:
                t = time.monotonic()
                await self.publish_state()
                self._phase("publish_state", t)

    async def _schedule_and_apply(self) -> None:
        rows = []
        t = time.monotonic()
        live = await self.store.live_leases()
        self._phase("live_leases", t)
        for row in live:
            if row["work_class"] not in self.cfg.classes or (row.get("role") and row["role"] not in self.cfg.roles):
                # The YAML no longer knows this class/role: end the lease rather than crash every tick.
                await self._resume(row["lease_id"], {"type": "cancel", "reason": "config_removed"})
                continue
            rows.append(row)
        t = time.monotonic()
        decisions = schedule(self.cfg, self.roles, self.cards, [self._view(r) for r in rows], self.now(),
                             seen_ctx=self._ctx_seen, guards=self.guard_states)
        self._phase("schedule", t)
        self._phases["decisions"] = self._phases.get("decisions", 0) + len(decisions)
        swaps: set[tuple] = set()
        for d in decisions:
            try:
                if isinstance(d, (SwapLoad, SwapUnload, SwapBlocked)):
                    swaps.add(self._swap_key(d))
                    await self._swap(d)
                    continue
                event: dict[str, Any] = {"type": _EVENT_FOR[type(d)], "reason": getattr(d, "reason", None)}
                if isinstance(d, Grant):
                    event["role"] = d.role
                if isinstance(d, Recall):
                    event["recall_by"] = d.recall_by.isoformat()
                await self._resume(d.lease_id, event)
            except Exception:  # noqa: BLE001 -- one bad decision must not block the rest
                logger.exception("gpu_pool_decision_failed decision=%s", d)
        # Edge-triggered: a swap decision is reported when it starts, and again if it recurs later.
        self._swap_requested &= swaps
        await self._touch_swap_seats(rows)

    @staticmethod
    def _swap_key(d: SwapLoad | SwapUnload | SwapBlocked) -> tuple:
        return (type(d).__name__, d.role, d.reason if isinstance(d, SwapBlocked) else None)

    async def _swap(self, d: SwapLoad | SwapUnload | SwapBlocked) -> None:
        """A seat in GPU_POOL_ACTUATE_ROLES is actuated; any other seat, and every blocked load, is
        reported as ``swap_requested {actuated: false}`` (edge-triggered: once when it starts, again
        if it recurs later)."""
        if isinstance(d, (SwapLoad, SwapUnload)) and d.role in self.actuate_roles:
            now = self.now()
            if not any(c.cooldown_until and c.cooldown_until > now for c in self._seat_cards(d.role)):
                await self._begin_actuation(d)
                return
            # Backoff after a refused/unanswered action. The scheduler already reports blocked
            # LOADS; an unload has no scheduler-side cooldown, and without this a dead actuator
            # would get a fresh unload every actuate_ack_sec.
            d = SwapBlocked(d.role, "cooldown", "unload" if isinstance(d, SwapUnload) else "load")
        key = self._swap_key(d)
        if key in self._swap_requested:
            return
        self._swap_requested.add(key)
        detail: dict[str, Any] = {"action": "unload" if isinstance(d, SwapUnload) else "load",
                                  "actuated": False, "mode": self.mode}
        if isinstance(d, SwapBlocked):
            detail.update(blocked=True, guard_state=d.detail)
        await self._emit(GpuPoolEventV1(
            event="swap_requested", role=d.role, cards=list(self.cfg.roles[d.role].cards), reason=d.reason,
            detail=detail))

    # --- actuation engine -----------------------------------------------------------------
    def _seat_cards(self, seat: str) -> list[CardLive]:
        return [self.cards[c] for c in self.cfg.roles[seat].cards]

    def _action_timeout(self, seat: str) -> float:
        """The whole action: a load drains+stops what it evicts then starts the seat; an unload
        stops the seat then starts the residents. Budget every launch it touches."""
        names = [seat, *self.cfg.evicted_by(seat)]
        total = sum(self.cfg.roles[r].launch.timeout_sec for r in names if self.cfg.roles[r].launch)
        return total or 900.0

    def _pending_actions(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for card in self.cards.values():
            if card.swap_state in ("loading", "unloading") and card.swap_action and card.swap_role \
                    and card.swap_role in self.cfg.roles:
                out.setdefault(card.swap_role, card.swap_action)
        return out

    async def _save_cards(self, cards: list[CardLive], by: str) -> None:
        for c in cards:
            await self.store.upsert_card({
                "card": c.card, "lent": c.lent, "swapped_in": sorted(c.swapped_in), "swap_state": c.swap_state,
                "cooldown_until": c.cooldown_until, "last_active_at": c.last_active_at, "swap_role": c.swap_role,
                "swap_generation": c.swap_generation, "swap_action": c.swap_action,
                "residency_until": c.residency_until, "loaded_at": c.loaded_at,
                "updated_at": self.now(), "updated_by": by})

    def _set_action(self, seat: str, **fields: Any) -> dict:
        cards = self._seat_cards(seat)
        act = {**(cards[0].swap_action or {}), **fields}
        for c in cards:
            c.swap_action = act
        return act

    async def _begin_actuation(self, d: SwapLoad | SwapUnload) -> None:
        seat = d.role
        spec = self.cfg.roles[seat]
        cards = self._seat_cards(seat)
        action = "load" if isinstance(d, SwapLoad) else "unload"
        now = self.now()
        generation = max(c.swap_generation for c in cards) + 1
        timeout = self._action_timeout(seat)
        msg = GpuActuateV1(
            action_id=f"{seat}:{action}:g{generation}:{uuid.uuid4().hex[:8]}", generation=generation,
            actuator=spec.launch.actuator, role=seat, action=action, cards=list(spec.cards), profile=None,
            launch_digest=launch_digest(self.cfg, seat), deadline_at=now + timedelta(seconds=timeout),
            reason=d.reason)
        act = {"action_id": msg.action_id, "role": seat, "action": action, "generation": generation,
               "reason": d.reason, "sent_at": now.isoformat(), "deadline_at": msg.deadline_at.isoformat(),
               "acked_at": None, "phase": None, "outcome": None, "status_action_id": None,
               "status_sent_at": None, "extensions": 0}
        for c in cards:
            c.swap_state = "loading" if action == "load" else "unloading"
            c.swap_role = seat
            c.swap_generation = generation
            c.swap_action = act
        # Persisted BEFORE the request leaves: a crash after the send still finds the action and
        # reconciles with `status` instead of sending a second transition.
        await self._save_cards(cards, "actuate")
        logger.info("gpu_pool_actuate_send seat=%s action=%s action_id=%s generation=%s reason=%s",
                    seat, action, msg.action_id, generation, d.reason)
        await self._emit(GpuPoolEventV1(event="swap_started", role=seat, cards=list(spec.cards), reason=d.reason,
                                        detail={"action": action, "action_id": msg.action_id,
                                                "generation": generation, "actuated": True}))
        await self._publish(GPU_POOL_ACTUATE_REQUEST_CHANNEL, GPU_ACTUATE_KIND, msg.model_dump(mode="json"), None)

    async def _send_status(self, seat: str, act: dict, *, reason: str) -> None:
        spec = self.cfg.roles[seat]
        now = self.now()
        status_id = f"{act.get('action_id')}:status:{uuid.uuid4().hex[:6]}"
        self._set_action(seat, status_action_id=status_id, status_sent_at=now.isoformat())
        await self._save_cards(self._seat_cards(seat), "actuate_status")
        msg = GpuActuateV1(
            action_id=status_id, generation=max(1, int(act.get("generation") or 1)),
            actuator=spec.launch.actuator if spec.launch else "unknown", role=seat, action="status",
            cards=list(spec.cards), profile=None, launch_digest=launch_digest(self.cfg, seat),
            deadline_at=now + timedelta(seconds=self.cfg.defaults.actuate_ack_sec), reason=reason)
        logger.info("gpu_pool_actuate_status seat=%s for=%s reason=%s", seat, act.get("action_id"), reason)
        await self._publish(GPU_POOL_ACTUATE_REQUEST_CHANNEL, GPU_ACTUATE_KIND, msg.model_dump(mode="json"), None)

    async def _finish(self, seat: str, *, state: str, loaded: bool | None, outcome: str,
                      event: str | None, reason: str | None, detail: dict | None = None) -> None:
        """Settle a seat's card set. ``loaded``: True puts the seat in swapped_in, False takes it out,
        None leaves swapped_in as it was (a refused/unanswered action changed nothing)."""
        now = self.now()
        cards = self._seat_cards(seat)
        prev_loaded = all(seat in c.swapped_in for c in cards)
        act = self._set_action(seat, outcome=outcome, finished_at=now.isoformat(), status_action_id=None)
        d = self.cfg.defaults
        for c in cards:
            c.swap_state = state
            c.swap_role = seat if state == "fault" else None
            if loaded is True:
                c.swapped_in.add(seat)
                if not prev_loaded or c.loaded_at is None:
                    c.loaded_at = now
                c.last_active_at = now   # idle unload counts from the load, not from before it
            elif loaded is False:
                c.swapped_in.discard(seat)
                c.loaded_at = None
                if prev_loaded:
                    c.residency_until = now + timedelta(seconds=d.swap_min_residency_sec)
            # Every way a load did not simply succeed backs off before the next attempt, including a
            # fault discovery cleared: otherwise a card whose rollback keeps half-failing flaps.
            if outcome in ("failed_restored", "refused", "unreachable", "fault_cleared", "stuck"):
                c.cooldown_until = now + timedelta(seconds=d.swap_cooldown_sec)
        await self._save_cards(cards, "actuate_result")
        log = logger.warning if state == "fault" or event in ("swap_failed", "actuate_refused") else logger.info
        log("gpu_pool_actuate_done seat=%s action_id=%s outcome=%s state=%s reason=%s",
            seat, act.get("action_id"), outcome, state, reason)
        if event:
            await self._emit(GpuPoolEventV1(
                event=event, role=seat, cards=list(self.cfg.roles[seat].cards), reason=reason,
                detail={"action": act.get("action"), "action_id": act.get("action_id"),
                        "generation": act.get("generation"), "swap_state": state, "actuated": True,
                        **(detail or {})}))

    def _observed_outcome(self, seat: str, observed: dict[str, str]) -> bool | None:
        """What the containers say: True = seat up and its evictees down, False = evictees up and
        seat down, None = neither (a half-done card is a fault, not a guess)."""
        residents = self.cfg.evicted_by(seat)
        seat_up = observed.get(seat) == "running"
        residents_up = [observed.get(r) == "running" for r in residents]
        if seat_up and not any(residents_up):
            return True
        if not seat_up and observed.get(seat) in ("exited", "absent") and all(residents_up):
            return False
        return None

    async def _adopt(self, seat: str, observed: dict[str, str], *, why: str) -> None:
        loaded = self._observed_outcome(seat, observed)
        if loaded is None:
            await self._finish(seat, state="fault", loaded=None, outcome="reconcile_ambiguous", event="swap_failed",
                               reason=f"reconcile_ambiguous:{why}", detail={"observed": observed})
        else:
            await self._finish(seat, state="idle", loaded=loaded, outcome="reconciled", event="swapped",
                               reason=f"reconciled:{why}", detail={"observed": observed, "loaded": loaded})

    async def on_actuate_result(self, res: GpuActuateResultV1) -> None:
        if not self._started:
            logger.warning("gpu_pool_actuate_result_before_start action_id=%s", res.action_id)
            return
        async with self._locked("actuate_result"):
            seat = res.role
            if seat not in self.cfg.roles or self.cfg.roles[seat].swap is None:
                logger.warning("gpu_pool_actuate_result_unknown_role role=%s action_id=%s", seat, res.action_id)
                return
            cards = self._seat_cards(seat)
            act = cards[0].swap_action or {}
            pending = cards[0].swap_state in ("loading", "unloading")
            now = self.now()
            if res.action == "status":
                if not pending or res.action_id != act.get("status_action_id"):
                    # Also the normal end of a reconcile: the actuator re-publishes the last result it
                    # recorded under ITS action_id first, which settled the card before this arrived.
                    logger.info("gpu_pool_actuate_status_stale seat=%s action_id=%s", seat, res.action_id)
                    return
                if res.status in ("accepted", "progress"):
                    return
                if res.status != "succeeded":
                    await self._finish(seat, state="fault", loaded=None, outcome=f"status_{res.status}",
                                       event="swap_failed", reason=f"status_{res.status}:{res.reason}")
                    return
                running = res.in_flight
                if running is None:   # an actuator that does not say: a reported phase means mid-action
                    running = res.phase is not None
                if running:
                    self._set_action(seat, status_action_id=None, phase=res.phase, in_flight=True,
                                     deadline_at=(now + timedelta(seconds=STATUS_POLL_SEC)).isoformat())
                    await self._save_cards(cards, "actuate_status")
                    logger.info("gpu_pool_actuate_still_running seat=%s action_id=%s phase=%s",
                                seat, act.get("action_id"), res.phase)
                    return
                observed = dict(res.observed)
                if self._observed_outcome(seat, observed) is None and res.in_flight is None:
                    n = int(act.get("extensions") or 0) + 1
                    if n <= MAX_STATUS_EXTENSIONS:
                        self._set_action(seat, status_action_id=None, extensions=n,
                                         deadline_at=(now + timedelta(seconds=STATUS_POLL_SEC)).isoformat())
                        await self._save_cards(cards, "actuate_status")
                        return
                # Nothing running there and our action's own result never came: the card is what the
                # containers say it is (or a fault when they say something half-done).
                await self._adopt(seat, observed, why="status")
                return

            if not pending or res.action_id != act.get("action_id"):
                # A result for an action we already gave up on (unanswered in time). If it is the
                # last one we issued and the card is idle, believe the containers it reports.
                terminal = res.status in ("succeeded", "failed")
                if terminal and res.action_id == act.get("action_id") and cards[0].swap_state == "idle" \
                        and res.observed and self._observed_outcome(seat, dict(res.observed)) is not None:
                    loaded = self._observed_outcome(seat, dict(res.observed))
                    if loaded != all(seat in c.swapped_in for c in cards):
                        await self._finish(seat, state="idle", loaded=loaded, outcome="late_result",
                                           event="swapped", reason="late_result",
                                           detail={"observed": dict(res.observed), "loaded": loaded})
                        return
                logger.info("gpu_pool_actuate_result_stale seat=%s action_id=%s status=%s",
                            seat, res.action_id, res.status)
                return

            if res.status in ("accepted", "progress"):
                self._set_action(seat, acked_at=act.get("acked_at") or now.isoformat(), phase=res.phase)
                await self._save_cards(cards, "actuate_progress")
                return
            action = act.get("action")
            if res.status == "succeeded":
                await self._finish(seat, state="idle", loaded=(action == "load"), outcome="succeeded",
                                   event="swapped", reason=act.get("reason"),
                                   detail={"elapsed_ms": res.elapsed_ms, "observed": dict(res.observed)})
            elif res.status == "refused":
                await self._finish(seat, state="idle", loaded=None, outcome="refused", event="actuate_refused",
                                   reason=res.reason or "refused")
            elif action == "load" and (res.restored is True or (
                    res.restored is None and self._observed_outcome(seat, dict(res.observed)) is False)):
                # restored=None: no rollback ran (nothing had been evicted yet); the containers say
                # whether the residents are still there.
                await self._finish(seat, state="idle", loaded=False, outcome="failed_restored",
                                   event="swap_failed", reason=res.reason or "failed",
                                   detail={"restored": res.restored, "phase": res.phase,
                                           "observed": dict(res.observed)})
            else:
                # A load that could not put the residents back, or an unload that failed: nobody
                # knows what the card holds. No grants on it until an operator or discovery clears it.
                await self._finish(seat, state="fault", loaded=None, outcome="failed", event="swap_failed",
                                   reason=res.reason or "failed",
                                   detail={"restored": res.restored, "phase": res.phase,
                                           "observed": dict(res.observed)})

    async def _check_actuation(self) -> None:
        """Timeouts (spec "Timeouts"): no accepted within actuate_ack_sec -> actuator_unreachable;
        no terminal result by deadline_at -> ask `status`; no answer to that -> fault."""
        now = self.now()
        ack = self.cfg.defaults.actuate_ack_sec
        for seat, act in self._pending_actions().items():
            if act.get("status_action_id"):
                sent = _ts(act.get("status_sent_at"))
                if sent and (now - sent).total_seconds() >= max(ack, STATUS_REPLY_SEC):
                    await self._finish(seat, state="fault", loaded=None, outcome="unreachable", event="swap_failed",
                                       reason="actuator_unreachable", detail={"during": "status"})
                continue
            sent = _ts(act.get("sent_at"))
            if not act.get("acked_at") and sent and (now - sent).total_seconds() >= ack:
                await self._finish(seat, state="idle", loaded=None, outcome="unreachable", event="swap_failed",
                                   reason="actuator_unreachable")
                continue
            deadline = _ts(act.get("deadline_at"))
            if deadline and now >= deadline:
                await self._send_status(seat, act, reason="deadline")

    async def _clear_faults(self) -> None:
        """fault -> idle once discovery sees a consistent card: the evicted residents healthy and
        the seat down (unloaded), or the seat confirmed up and its residents down (loaded)."""
        for card in list(self.cards.values()):
            seat = card.swap_role
            if card.swap_state != "fault" or not seat or seat not in self.cfg.roles:
                continue
            residents = self.cfg.evicted_by(seat)
            up = [bool(self.probes.get(r) and self.probes[r].ok) for r in residents]
            alive = self._seat_alive(seat)
            if all(up) and not alive:
                loaded = False
            elif alive and not any(up):
                loaded = True
            else:
                continue
            await self._finish(seat, state="idle", loaded=loaded, outcome="fault_cleared", event="swapped",
                               reason="fault_cleared:discovery", detail={"loaded": loaded})

    async def _touch_swap_seats(self, rows: list[dict]) -> None:
        now = self.now()
        for row in rows:
            role = row.get("role")
            if row["status"] in ("granted", "recalling") and role and self.cfg.roles[role].swap:
                for card in self.cfg.roles[role].cards:
                    self.cards[card].last_active_at = now

    # --- lease-graph plumbing ---------------------------------------------------------
    async def _start_thread(self, lease_id: str, request: dict, now: datetime,
                            parent_lease_id: str | None = None) -> None:
        state = initial_state(lease_id, request, now)
        await self.graph.ainvoke(state, self._thread(lease_id))
        row = self._row(state, request, parent_lease_id)
        await self.store.upsert_lease(row)
        await self._emit_row("admitted", row)

    async def _resume(self, lease_id: str, event: dict[str, Any]) -> dict | None:
        t = time.monotonic()
        try:
            return await self._resume_inner(lease_id, event)
        finally:
            self._phase("resume", t)

    async def _resume_inner(self, lease_id: str, event: dict[str, Any]) -> dict | None:
        from langgraph.types import Command

        cfg = self._thread(lease_id)
        snap = await self.graph.aget_state(cfg)
        if not snap.values or not snap.next:
            return None
        event = {**event, "at": self.now().isoformat()}
        try:
            values = await self.graph.ainvoke(Command(resume=event), cfg)
        except Exception as exc:  # noqa: BLE001 -- LangGraph may wrap node errors
            if isinstance(exc, InvalidTransition) or isinstance(exc.__cause__, InvalidTransition) \
                    or "-/->" in str(exc):
                logger.info("gpu_pool_transition_rejected lease=%s %s", lease_id, exc)
                await self._sync_row(lease_id)  # heal a projection that fell behind its checkpoint
                return None
            raise
        prior = await self.store.lease(lease_id)
        row = self._row(values, values["request"], (prior or {}).get("parent_lease_id"))
        await self.store.upsert_lease(row)
        # A lease granted on, or leaving, a swap seat is activity there: the idle-unload clock must
        # see a lease that came and went between two ticks, not only what a tick happened to catch.
        seat = row.get("role") or (prior or {}).get("role")
        if seat in self.cfg.roles and self.cfg.roles[seat].swap is not None \
                and (row["status"] in ("granted", "recalling") or (prior or {}).get("status") in ("granted", "recalling")):
            for card in self.cfg.roles[seat].cards:
                self.cards[card].last_active_at = self.now()
        # What happened (aborted / expired / cancelled), then where it ended up when that is a
        # terminal outcome (dead_lettered / released), so neither fact hides the other.
        names: list[str] = []
        if event["type"] == "abort":
            names.append("aborted")
        elif event["type"] == "expire":
            names.append("expired")
        elif event["type"] == "cancel":
            names.append("cancelled")
        if event["type"] != "heartbeat" and (not names or row["status"] == "dead_letter"
                                             or (row["status"] == "released" and names != ["cancelled"])):
            outcome = _PUBLIC.get(row["status"])
            if outcome and outcome not in names:
                names.append(outcome)
        for name in names:
            await self._emit_row(name, row, prior=prior, reason=row.get("reason") or event.get("reason"))
        return row

    async def _sync_row(self, lease_id: str) -> dict | None:
        """The checkpoint is the truth; rewrite the projection row from it if they disagree
        (e.g. a crash between the graph commit and the row upsert)."""
        snap = await self.graph.aget_state(self._thread(lease_id))
        values = snap.values or {}
        if not values:
            return None
        prior = await self.store.lease(lease_id)
        if prior is not None and prior["status"] == values["status"] and prior.get("role") == values.get("role"):
            return prior
        row = self._row(values, values["request"], (prior or {}).get("parent_lease_id"))
        await self.store.upsert_lease(row)
        logger.warning("gpu_pool_projection_healed lease=%s %s->%s", lease_id,
                       (prior or {}).get("status"), row["status"])
        return row

    def _row(self, values: dict, request: dict, parent_lease_id: str | None) -> dict[str, Any]:
        return {
            "lease_id": values["lease_id"], "request_id": request["request_id"],
            "holder": request.get("holder") or "unknown", "work_class": request["work_class"],
            "priority": request.get("priority", "system"), "kind": request.get("kind", "request"),
            "status": values["status"], "role": values.get("role"), "attempt": values.get("attempt", 1),
            "generation": values.get("generation", 0), "operator": bool(request.get("operator")),
            "min_ctx_tokens": int(request.get("min_ctx_tokens") or 0),
            "needs_vision": bool(request.get("needs_vision")),
            "retryable": bool(request.get("retryable")),
            "created_at": _ts(values["created_at"]), "queued_since": _ts(values.get("queued_since")),
            "granted_at": _ts(values.get("granted_at")), "recall_by": _ts(values.get("recall_by")),
            "not_before": _ts(values.get("not_before")), "deadline_at": _ts(request.get("deadline_at")),
            "expires_at": _ts(values.get("expires_at")),
            "turn_correlation_id": request.get("turn_correlation_id"),
            "parent_lease_id": parent_lease_id or request.get("parent_lease_id"),
            "hold_lease_id": request.get("hold_lease_id"),
            "reason": values.get("reason"), "updated_at": self.now(),
        }

    @staticmethod
    def _view(row: dict) -> LeaseView:
        return LeaseView(
            lease_id=row["lease_id"], work_class=row["work_class"], priority=row["priority"],
            status=row["status"], created_at=row["created_at"], role=row.get("role"),
            min_ctx_tokens=row.get("min_ctx_tokens") or 0, needs_vision=bool(row.get("needs_vision")),
            deadline_at=row.get("deadline_at"), recall_by=row.get("recall_by"),
            not_before=row.get("not_before"), queued_since=row.get("queued_since"),
            granted_at=row.get("granted_at"), expires_at=row.get("expires_at"),
            operator=bool(row.get("operator")), retryable=bool(row.get("retryable")),
            kind=row.get("kind") or "request", hold_lease_id=row.get("hold_lease_id"),
        )

    # --- replies ------------------------------------------------------------------------
    def grant_for(self, role: str, lease_id: str, generation: int) -> GpuLeaseGrantV1:
        disc = next((d for d in self.discovered if d.role == role), None)
        return GpuLeaseGrantV1(
            lease_id=lease_id, generation=max(1, generation), role=role,
            cards=list(self.cfg.roles[role].cards), url=self.cfg.url(role),
            profile_name=disc.profile_name if disc else None, model_file=disc.model_file if disc else None,
            ctx_per_slot=disc.ctx_per_slot if disc else None,
            # "{node}-worker-{role}": cortex-exec reads the node before "-worker" to attribute
            # reasoning_load (executor._normalize_served_by_to_node); any other shape loses it.
            served_by=f"{self.cfg.host.name}-worker-{role}")

    async def _reply_for(self, row: dict) -> GpuLeaseReplyV1:
        status = row["status"]
        if status == "granted":
            return GpuLeaseReplyV1(status="granted", lease_id=row["lease_id"],
                                   grant=self.grant_for(row["role"], row["lease_id"], row["generation"]))
        if status == "recalling":
            return GpuLeaseReplyV1(status="recall", lease_id=row["lease_id"], recall_by=row["recall_by"],
                                   grant=self.grant_for(row["role"], row["lease_id"], row["generation"]))
        if status in ("queued", "retry_wait"):
            return GpuLeaseReplyV1(status="queued", lease_id=row["lease_id"], position=await self._position(row))
        if status == "backlogged":
            return GpuLeaseReplyV1(status="backlogged", lease_id=row["lease_id"], reason=row.get("reason"))
        if status == "released":
            return GpuLeaseReplyV1(status="ok", lease_id=row["lease_id"], reason=row.get("reason"))
        return GpuLeaseReplyV1(status="unavailable", lease_id=row["lease_id"], reason=row.get("reason") or status)

    async def _position(self, row: dict) -> int:
        rank = self.cfg.priority_rank
        mine = (rank(row["priority"]), row["created_at"])
        return 1 + sum(1 for r in await self.store.live_leases()
                       if r["status"] == "queued" and r["work_class"] == row["work_class"]
                       and (rank(r["priority"]), r["created_at"]) < mine)

    # --- state + events ---------------------------------------------------------------
    async def snapshot(self, include_leases: bool = True, include_config: bool = False,
                       history_for: str | None = None) -> GpuPoolStateV1:
        rows = await self.store.live_leases()
        queue: dict[str, int] = {}
        backlog: dict[str, int] = {}
        for r in rows:
            if r["status"] == "queued":
                queue[r["work_class"]] = queue.get(r["work_class"], 0) + 1
            elif r["status"] == "backlogged":
                backlog[r["work_class"]] = backlog.get(r["work_class"], 0) + 1
        return GpuPoolStateV1(
            mode="enforce" if self.mode == "enforce" else "observe", config_digest=self.cfg.digest,
            cards=[GpuCardStateV1(card=c.card, vram_gb=self.cfg.cards[c.card].vram_gb,
                                  lendable=self.cfg.cards[c.card].lendable, lent=c.lent,
                                  swapped_in=sorted(c.swapped_in), swap_state=c.swap_state,
                                  cooldown_until=c.cooldown_until, swap_role=c.swap_role,
                                  residency_until=c.residency_until, loaded_at=c.loaded_at,
                                  actuated_roles=sorted(r for r in self.actuate_roles
                                                        if c.card in self.cfg.roles[r].cards),
                                  actuation=c.swap_action) for c in self.cards.values()],
            roles=self.discovered, unclaimed_servers=self.unclaimed,
            leases=[GpuLeaseRowV1(
                lease_id=r["lease_id"], request_id=r["request_id"], holder=r["holder"],
                work_class=r["work_class"], priority=r["priority"], kind=r["kind"], status=r["status"],
                role=r.get("role"), attempt=r.get("attempt", 1), created_at=r["created_at"],
                granted_at=r.get("granted_at"), recall_by=r.get("recall_by"),
                turn_correlation_id=r.get("turn_correlation_id"), generation=int(r.get("generation") or 0),
                hold_lease_id=r.get("hold_lease_id")) for r in rows] if include_leases else [],
            queue_depth=queue, backlog_depth=backlog, swap_guards=dict(self.guard_states),
            config=self.cfg.model_dump(mode="json", by_alias=True, exclude={"digest"}) if include_config else None,
            config_yaml=self.cfg.source_text if include_config else None,
            history_lease_id=history_for,
            history=await self.history(history_for) if history_for else None)

    async def publish_state(self) -> None:
        self._last_state = self.now()
        if self.bus is None:
            return
        state = await self.snapshot()
        await self._publish(GPU_POOL_STATE_CHANNEL, GPU_POOL_STATE_KIND, state.model_dump(mode="json"), None)

    async def history(self, lease_id: str) -> list[dict[str, Any]]:
        """The walker's path: the lease's own history as its last checkpoint recorded it."""
        snap = await self.graph.aget_state(self._thread(lease_id))
        return list((snap.values or {}).get("history") or [])

    async def _emit_row(self, event: str, row: dict, *, prior: dict | None = None, reason: str | None = None) -> None:
        waited_ms = held_ms = None
        now = self.now()
        if event == "granted" and row.get("queued_since"):
            waited_ms = (now - row["queued_since"]).total_seconds() * 1000
        if event in ("released", "aborted", "expired", "retried") and prior and prior.get("granted_at"):
            held_ms = (now - prior["granted_at"]).total_seconds() * 1000
        detail: dict[str, Any] = {}
        if event == "granted" and row.get("role"):
            detail["grant"] = self.grant_for(row["role"], row["lease_id"], row["generation"]).model_dump(mode="json")
        if event == "recalled":
            detail["recall_by"] = row["recall_by"].isoformat() if row.get("recall_by") else None
        await self._emit(GpuPoolEventV1(
            event=event, lease_id=row["lease_id"], holder=row["holder"], work_class=row["work_class"],
            priority=row["priority"], role=row.get("role") or (prior or {}).get("role"),
            cards=list(self.cfg.roles[row["role"]].cards) if row.get("role") else [],
            turn_correlation_id=row.get("turn_correlation_id"), attempt=row.get("attempt"),
            waited_ms=waited_ms, held_ms=held_ms, reason=reason or row.get("reason"), detail=detail))
        if self.bus is not None:
            hop = f"gpu_pool:{row['work_class']}#{WAIT_HOP_LABEL}"
            if event == "granted" and waited_ms is not None:
                self.bus.record_hop_success(hop, waited_ms)
            elif event == "unavailable" and (reason or row.get("reason")) == "deadline":
                self.bus.record_hop_timeout(hop, None)

    async def _emit(self, event: GpuPoolEventV1) -> None:
        if self.bus is None:
            return
        await self._publish(GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND, event.model_dump(mode="json"),
                            event.turn_correlation_id)
        if event.event in _GRAMMAR_EVENTS:
            await self._grammar(event)

    async def _publish(self, channel: str, kind: str, payload: dict, corr: str | None) -> None:
        t = time.monotonic()
        try:
            await self._publish_inner(channel, kind, payload, corr)
        finally:
            self._phase("bus_publish", t)

    async def _publish_inner(self, channel: str, kind: str, payload: dict, corr: str | None) -> None:
        try:
            cid = uuid.UUID(str(corr)) if corr else uuid.uuid4()
        except ValueError:
            cid = uuid.uuid4()
        try:
            await self.bus.publish(channel, BaseEnvelope(
                kind=kind, source=ServiceRef(name=self.service_name), correlation_id=cid, payload=payload))
        except Exception:  # noqa: BLE001 -- telemetry must never break scheduling
            logger.warning("gpu_pool_publish_failed channel=%s", channel, exc_info=True)

    async def _grammar(self, event: GpuPoolEventV1) -> None:
        """Lease facts land in grammar_events via sql-writer. Layer "capacity", never "transport":
        waiting in line is not transport. No reducer reads these until the metric gate (stage 6)."""
        try:
            from orion.grammar.publish import publish_grammar_event
            from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1

            trace_id = f"gpu_pool.lease:{event.lease_id or event.role or 'pool'}"
            event_id = f"{trace_id}:{event.event_id[:12]}"
            dims = ["capacity", "gpu", event.work_class or "pool"]
            atom = GrammarAtomV1(
                atom_id=event_id, trace_id=trace_id, atom_type="observation",
                semantic_role=f"gpu_lease_{event.event}", layer="capacity", dimensions=dims,
                summary=(f"{event.event} class={event.work_class} role={event.role} "
                         f"waited_ms={event.waited_ms} reason={event.reason}"),
                text_value=event.role or event.work_class or "pool", confidence=1.0, salience=0.5,
                source_event_id=event.event_id)
            await publish_grammar_event(self.bus, GrammarEventV1(
                event_id=event_id, event_kind="atom_emitted", trace_id=trace_id,
                correlation_id=event.turn_correlation_id, emitted_at=event.generated_at, layer="capacity",
                dimensions=dims, atom=atom,
                provenance=GrammarProvenanceV1(source_service=self.service_name, source_component="lease_graph",
                                               source_event_id=event.event_id, source_trace_id=trace_id)),
                source_name=self.service_name)
        except Exception:  # noqa: BLE001
            logger.warning("gpu_pool_grammar_failed event=%s", event.event, exc_info=True)
