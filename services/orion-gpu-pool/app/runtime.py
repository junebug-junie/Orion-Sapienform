"""The pool runtime: the single writer that joins discovery, the scheduler, the lease graph,
the fenced projection and the bus.

Every entrypoint (RPC verb, announcement, tick) runs under one asyncio lock, so decisions are
serial and the scheduler always sees a consistent world. Only the Postgres advisory-lock holder
runs a runtime at all (see app/main.py), so this lock is the whole concurrency story.

Stage 4.3 adds durable-run holds (acquire kind=hold, ``attach`` for each call under one, ``status``
for resume) and the swap actuation engine: for a swap seat with a ``launch`` block the pool sends
GpuActuateV1 to the seat's host actuator and walks the card through
idle -> loading|unloading -> idle|fault from the GpuActuateResultV1 replies, persisting every step
on gpu_pool_cards so a restart reconciles (``status``) instead of issuing a second transition.

Stage 5.7 (enforce is the end state): a seat is actuated iff it has a launch block (the
GPU_POOL_ACTUATE_ROLES list is deleted). The one emergency stop is the ``pause_actuation`` control
verb, persisted on gpu_pool_cards: while paused every swap decision is published as
``swap_requested {actuated: false, paused: true}`` and no seat is drained. In enforce mode a boot (and
a resume) asks the actuator what each idle seat's cards really hold and adopts it.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.gpu_pool.client import WAIT_HOP_LABEL
from orion.gpu_pool.config import PoolConfig, launch_digest
from orion.gpu_pool.discovery import Probe, resolve_roles
from orion.gpu_pool.lease_graph import FINAL, InvalidTransition, initial_state
from orion.gpu_pool.scheduler import (
    Abort, Backlog, CardLive, DeadLetter, Expire, Grant, LeaseView, Recall, Requeue, RoleLive,
    Serialized, Shed, SwapBlocked, SwapLoad, SwapUnload, Unavailable, hold_cap, schedule,
)
from orion.gpu_pool.orion_shed import (
    SHED_DECISION_REASON as ORION_SHED_DECISION_REASON, MemoryOrionShedLedger, OrionShedCaps,
    OrionShedController, OrionShedLedger,
)
from orion.gpu_pool.shed import REFLEX_REASONS, ShedBoard, ShedSignal
from orion.schemas.hardware_watch import HardwareWatchIncidentV1, HardwareWatchReflexShedV1
from orion.gpu_pool.config import SWAP_GUARDS
from orion.gpu_pool.controller_health import ControllerHealth
from orion.schemas.gpu_pool import (
    GPU_ACTUATE_KIND, GPU_POOL_ACTUATE_REQUEST_CHANNEL, GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND,
    GPU_POOL_STATE_CHANNEL, GPU_POOL_STATE_KIND,
    DiscoveredRoleV1, GpuActuateResultV1, GpuActuateV1, GpuCardStateV1, GpuLeaseGrantV1, GpuLeaseReplyV1,
    GpuLeaseRequestV1, GpuLeaseRowV1, GpuPoolControlReplyV1, GpuPoolControlV1, GpuPoolEventV1,
    GpuPoolShedReasonRequestV1, GpuPoolShedResultV1, GpuPoolStateV1, LlmWorkerAnnounceV1,
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


# A reflex signal (v2 cabinet_hot/cabinet_unknown, v1 cooling_incident) lasts until the producer's
# valid_until (v2: 3 watcher ticks; v1 default 300 s), never longer than SHED_MAX_VALID_SEC from when
# the pool received it.
SHED_DEFAULT_VALID_SEC = 300.0
V2_REFLEX_REASONS = ("cabinet_hot", "cabinet_unknown")
SHED_MAX_VALID_SEC = 900.0


def event_envelope_correlation(event: GpuPoolEventV1) -> uuid.UUID:
    """The bus envelope correlation id for one pool event: its own ``event_id`` (a uuid4 hex),
    fresh per event and joinable to gpu_pool_events.event_id. Never the turn's id (see ``_emit``)."""
    try:
        return uuid.UUID(hex=str(event.event_id))
    except ValueError:
        return uuid.uuid4()


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
# Even while the actuator keeps saying "running", a card is faulted this many action timeouts after
# the action was sent: an actuator wedged mid-transition must not hold a card in `loading` forever.
MAX_ACTION_TIMEOUTS = 2.0
# Actions this process issued per seat, newest last: a late result for any of them is still ours.
RECENT_ACTIONS = 8
# A status reply that cannot say whether the action is running (an actuator predating the
# `in_flight` field) and shows a half-done card is re-asked this many times before the card faults.
MAX_STATUS_EXTENSIONS = 4


POOL_MODES = ("enforce", "observe")


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
                 mode: str = "enforce", service_name: str = "orion-gpu-pool",
                 announce_stale_sec: float = 120.0, probe_interval_sec: float = 15.0,
                 state_publish_sec: float = 5.0, replay_payload_max_bytes: int = 262144,
                 shed_enabled: bool = False, orion_shed_enabled: bool = False,
                 orion_shed_caps: OrionShedCaps | None = None, orion_shed_ledger: OrionShedLedger | None = None,
                 controller_alert: Callable[[str, str, dict[str, Any]], Awaitable[Any]] | None = None):
        if mode not in POOL_MODES:
            raise ValueError(f"GPU_POOL_MODE must be one of {POOL_MODES}, got {mode!r}")

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
        self._serialized_reported: set[tuple] = set()   # (lease_id, reason) already reported
        self._hold_cap_reported: dict[str, str | None] = {}   # role -> last logged clamp reason
        self._ctx_seen: dict[str, int] = {}
        # Stage 5.7: config, not a list. A seat is actuated iff it has a launch block.
        self.actuated = cfg.actuated_seats()
        # The emergency stop (control verb pause_actuation), persisted on every gpu_pool_cards row:
        # {"since": datetime, "by": actor} while paused, else None.
        self.paused: dict[str, Any] | None = None
        # enforce: seat -> the `status` this process sent to learn what the seat's cards really hold
        # (boot and resume). Nothing is actuated on that seat until it is answered or times out.
        self._reconciling: dict[str, dict[str, Any]] = {}
        # Swap-load guards (app/guards.py fills these outside the lock). Every guard starts failing
        # ("unread"): a pool that has not read the thermal sensor yet must not load a model.
        self.guard_states: dict[str, str | None] = {g: "unread" for g in SWAP_GUARDS}
        # U4 shed lever (orion/gpu_pool/shed.py). GPU_POOL_SHED_ENABLED=false keeps every signal
        # visible but blocks nothing.
        self.shed_board = ShedBoard()
        self.shed_enabled = shed_enabled
        self._shed_reported: set[tuple] = set()   # (lease_id, reason) already reported
        self._shed_active: str | None = None      # last logged active reason (edge-triggered log)
        # Orion's learned shed (attend-to-act A1): the lower-precedence ``orion_self_shed`` reason, its
        # caps, ledger and manipulation check. GPU_POOL_ORION_SHED_ENABLED=false refuses every set.
        # (thermal controller v2, D2: the old _open_ac_incidents set is gone with its no-expiry bug, C12.
        # The learned action is refused while a REFLEX reason is active on the board, which expires.)
        self.orion_shed = OrionShedController(
            board=self.shed_board, ledger=orion_shed_ledger or MemoryOrionShedLedger(),
            caps=orion_shed_caps or OrionShedCaps(), enabled=orion_shed_enabled,
            lever_enabled=lambda: self.shed_enabled, now=lambda: self.now())
        # Can each seat's actuator act on what the pool asks? (orion/gpu_pool/controller_health.py; the
        # 2026-10-09 stale-controller incident.) controller_alert(seat, "degraded"|"recovered", view)
        # is fired once per transition, outside the lock (main.py wires it to a Hub attention card).
        self.controller_health = ControllerHealth()
        self.controller_alert = controller_alert
        self._alert_tasks: set[asyncio.Task] = set()
        self._started = False
        self._recent_actions: dict[str, list[str]] = {}
        self._ctx_saved: dict[str, int] = {}

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
            for role, size in (_action((row or {}).get("seen_ctx")) or {}).items():
                if role in self.cfg.roles and size:
                    self._ctx_seen.setdefault(role, int(size))
        self._ctx_saved = dict(self._ctx_seen)
        # Only configured cards: a row for a card since removed from the YAML must not pause the pool.
        paused = [r for c, r in stored.items() if c in self.cfg.cards and r.get("actuation_paused_at")]
        if paused:
            first = min(paused, key=lambda r: _ts(r["actuation_paused_at"]))
            self.paused = {"since": _ts(first["actuation_paused_at"]), "by": first.get("actuation_paused_by")}
            logger.warning("gpu_pool_actuation_paused_at_boot since=%s by=%s -- no model is loaded or unloaded "
                           "until verb=resume_actuation", self.paused["since"].isoformat(), self.paused["by"])
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
        await self._reconcile_idle_seats("boot")
        for did in await self.orion_shed.boot():
            logger.warning("gpu_pool_orion_shed_boot %s enabled=%s", did, self.orion_shed.enabled)

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
        if seat not in self.actuated:
            return False
        cards = [self.cards[c] for c in self.cfg.roles[seat].cards]
        return any(c.swap_generation > 0 or c.swap_state != "idle" for c in cards)

    def _observe_swap_seats(self) -> None:
        """observe mode only (the stage 5.7 rollback): a swap seat is marked loaded when its worker
        is really up, until the pool's first action there. That is how a seat loaded by hand is
        adopted without a reload. After the pool's first action its own intent is the truth, and
        discovery still gates grants on the worker being confirmed. enforce mode never guesses from
        liveness: it asks the actuator (``_reconcile_idle_seats``, at boot and on resume)."""
        if self.mode == "enforce":
            return
        for role, spec in self.cfg.roles.items():
            if spec.swap is None or self._pool_owns(role):
                continue
            alive = self._seat_alive(role)
            for card in spec.cards:
                c = self.cards[card]
                if alive:
                    if role in self.actuated and role not in c.swapped_in:
                        # Adopted: its max_hold_sec and idle-unload clocks start now. Without the
                        # second, an armed pool would unload a 27B the old path just loaded on
                        # its first tick, before the run that asked for it could be granted.
                        c.loaded_at = self.now()
                        c.last_active_at = self.now()
                    c.swapped_in.add(role)
                else:
                    if role in c.swapped_in and role in self.actuated:
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
            if operator and (why := self._operator_refusal(req.work_class)):
                return GpuLeaseReplyV1(status="unavailable", reason=why)
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

    def _operator_refusal(self, work_class: str) -> str | None:
        """Why an operator lease must not be admitted now, most permanent reason first (read under the
        runtime lock, so a pause cannot land between the check and the admit). A seat with no launch
        block can never load (stage 5.7: its lease would only drain residents or sit forever); observe
        mode (the rollback) keeps operator holds off; a paused pool cannot load the seat."""
        if why := self.cfg.not_actuatable_reason(work_class):
            return why
        if self.mode != "enforce":
            return "hold_refused_observe_mode"
        if self.paused is not None:
            return "actuation_paused"
        return None

    async def attach(self, req: GpuLeaseRequestV1) -> GpuLeaseReplyV1:
        """A call made under a durable run's hold: a request lease that runs in the hold's slot
        (H2), idempotent on request_id. The hold must be live at the generation the caller holds:
        a re-granted hold means the caller's view is stale and it must not attach."""
        async with self._locked("attach"):
            existing = await self.store.lease_by_request(req.request_id)
            if existing is not None:
                if existing.get("hold_lease_id") != req.hold_lease_id:
                    # A request_id that names some OTHER lease (e.g. the hold's own): never hand
                    # that lease back -- the caller would release/cancel it on exit.
                    return GpuLeaseReplyV1(status="unavailable", reason="request_id_conflict")
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
                       # never retryable: a re-grant after the caller left would sit in the run's slot
                       "kind": "request", "retryable": False, "holder": req.holder or hold["holder"],
                       "operator": False}
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
        if ctl.verb in ("pause_actuation", "resume_actuation"):
            return await self._set_paused(ctl.verb == "pause_actuation", ctl.actor)
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
        if ctl.verb == "clear_fault":
            return await self._clear_fault(ctl)
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
            self._report_hold_caps()
            for d in self._discovery_changes:
                await self._emit(GpuPoolEventV1(
                    event="discovery_confirmed" if d.status == "confirmed" else "discovery_mismatch",
                    role=d.role, cards=d.cards, reason=d.detail,
                    detail={"profile_name": d.profile_name, "model_file": d.model_file}))
            await self._save_seen_ctx()
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
        try:
            await self.orion_shed.expire_due()
        except Exception:  # noqa: BLE001 -- bookkeeping must never stop the pool
            logger.exception("gpu_pool_orion_shed_expire_failed")
        shed = self.shed_view()
        t = time.monotonic()
        decisions = schedule(self.cfg, self.roles, self.cards, [self._view(r) for r in rows], self.now(),
                             seen_ctx=self._ctx_seen, guards=self.guard_states, shed=shed.blocked,
                             frozen=(self.actuated if self.paused is not None else frozenset()) | set(self._reconciling))
        self._phase("schedule", t)
        self._phases["decisions"] = self._phases.get("decisions", 0) + len(decisions)
        swaps: set[tuple] = set()
        serialized: set[tuple] = set()
        shed_now: set[tuple] = set()
        by_id = {r["lease_id"]: r for r in rows}
        for d in decisions:
            try:
                if isinstance(d, Shed):
                    shed_now.add((d.lease_id, d.reason))
                    await self._report_shed(d, by_id.get(d.lease_id), shed)
                    continue
                if isinstance(d, Serialized):
                    serialized.add((d.lease_id, d.reason))
                    await self._report_serialized(d, by_id.get(d.lease_id))
                    continue
                if isinstance(d, (SwapLoad, SwapUnload, SwapBlocked)):
                    # Keep the key _swap actually reported under: it may rewrite the decision
                    # (a cooled-down unload becomes SwapBlocked(cooldown)). Keeping the scheduler's
                    # own key here wiped that memory every tick -- 628 events in 10 min on 09-30.
                    swaps.add(await self._swap(d))
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
        self._serialized_reported &= serialized
        self._shed_reported &= shed_now
        try:
            await self.orion_shed.on_tick(
                live_rows=rows, withheld_lease_ids={lid for lid, why in shed_now if why == ORION_SHED_DECISION_REASON})
        except Exception:  # noqa: BLE001 -- bookkeeping must never stop the pool
            logger.exception("gpu_pool_orion_shed_tick_failed")
        await self._touch_swap_seats(rows)

    # --- U4 shed ------------------------------------------------------------------------
    def on_incident(self, ev: HardwareWatchIncidentV1) -> str:
        """v1 rollback path (HARDWARE_WATCH_HEAT_CONTROLLER=v1): a cooling incident -> the
        ``cooling_incident`` shed signal for that incident. Open + shed.requested sets (or refreshes)
        it; anything else clears it. A v2 watcher's cooling incidents carry ``shed=None`` and only
        clear. The signal lapses at ``shed.valid_until``, capped at SHED_MAX_VALID_SEC from now so a
        bad clock can never latch shedding. Returns what it did, for the log."""
        if ev.rule != "cooling":
            return "ignored"
        if ev.status != "open" or ev.shed is None or not ev.shed.requested:
            return "cleared" if self.shed_board.clear("cooling_incident", ev.incident_id) else "noop"
        now = self.now()
        cap = now + timedelta(seconds=SHED_MAX_VALID_SEC)
        valid_until = min(ev.shed.valid_until or (now + timedelta(seconds=SHED_DEFAULT_VALID_SEC)), cap)
        self.shed_board.set(ShedSignal(
            "cooling_incident", ev.incident_id, ev.shed.requested_at or ev.opened_at, valid_until,
            {"subject": ev.subject, "open_reason": ev.open_reason, "shed_reason": ev.shed.reason,
             "cabinet_temp_c": ev.shed.cabinet_temp_c}))
        return "set"

    async def handle_incident(self, ev: HardwareWatchIncidentV1) -> str:
        """on_incident + the learned action's rule: when the reflex starts shedding (v1: a cooling
        incident sets ``cooling_incident``), any active ``orion_self_shed`` ends at once (settles
        ``preempted_by_reflex``). An incident that does not shed (any v2 incident, a heat incident)
        no longer touches the learned action (C7)."""
        async with self._locked("incident"):
            did = self.on_incident(ev)
            if did == "set" and await self.orion_shed.preempt_by_reflex(f"cooling_incident:{ev.incident_id}"):
                did = f"{did}+orion_shed_preempted"
            return did

    def on_reflex_shed(self, sig: HardwareWatchReflexShedV1) -> str:
        """v2 reflex (D2): one tick of hardware-watch's ``cabinet_hot``/``cabinet_unknown`` claim.
        Active sets (or refreshes) that reason for ``source_id`` and clears the other reflex reason
        from the same source; inactive clears both. Capped at SHED_MAX_VALID_SEC from now."""
        cleared = False
        for reason in V2_REFLEX_REASONS:
            if not sig.active or reason != sig.reason:
                cleared = self.shed_board.clear(reason, sig.source_id) or cleared
        if not sig.active:
            return "cleared" if cleared else "noop"
        now = self.now()
        valid_until = min(sig.valid_until, now + timedelta(seconds=SHED_MAX_VALID_SEC))
        if valid_until <= now:
            return "expired_on_arrival"
        fresh = not any(r["name"] == sig.reason and r["active"] for r in self.shed_board.view(now, True).reasons)
        self.shed_board.set(ShedSignal(sig.reason, sig.source_id, now, valid_until, {
            k: sig.cabinet.get(k) for k in ("temp_c", "thermal_state", "age_sec", "critical", "ac_low")}))
        return "set" if fresh else "refreshed"

    async def handle_reflex_shed(self, sig: HardwareWatchReflexShedV1) -> str:
        """on_reflex_shed + the learned action's rule: a reflex reason starting preempts it."""
        async with self._locked("reflex_shed"):
            did = self.on_reflex_shed(sig)
            if did == "set" and await self.orion_shed.preempt_by_reflex(f"{sig.reason}:{sig.source_id}"):
                did = f"{did}+orion_shed_preempted"
            return did

    def reflex_active(self) -> str | None:
        """The active reflex reason on the board (any precedence-0 hardware-watch reason), or None."""
        view = self.shed_board.view(self.now(), True)
        return next((r["name"] for r in view.reasons if r["name"] in REFLEX_REASONS and r["active"]), None)

    async def orion_shed_request(self, req: GpuPoolShedReasonRequestV1) -> GpuPoolShedResultV1:
        """The shed RPC. Under the runtime lock: it reads live leases and changes the board."""
        async with self._locked("orion_shed"):
            live = await self.store.live_leases()
            background_live = sum(1 for r in live if r.get("priority") == "background")
            return await self.orion_shed.handle(req, background_live=background_live,
                                                reflex_active=self.reflex_active() is not None)

    def shed_view(self):
        """The board as the scheduler will use it now; logs when the active reason changes."""
        now = self.now()
        for gone in self.shed_board.prune(now):
            logger.warning("gpu_pool_shed_lapsed reason=%s source=%s valid_until=%s -- no refresh from its producer",
                           gone.reason, gone.source_id, gone.valid_until.isoformat())
        view = self.shed_board.view(now, self.shed_enabled)
        active = next((r["name"] for r in view.reasons if r["active"]), None)
        if active != self._shed_active:
            logger.warning("gpu_pool_shed_%s reason=%s enabled=%s blocked=%s", "on" if active else "off",
                           active or self._shed_active, self.shed_enabled, view.blocked)
            self._shed_active = active
        return view

    async def _report_shed(self, d: Shed, row: dict | None, view) -> None:
        """U4: a ``queued`` event naming the shed reason (and the incident behind it), once per
        lease and reason while it lasts. Not a lease transition: the row stays where it is."""
        key = (d.lease_id, d.reason)
        if key in self._shed_reported or row is None:
            return
        name = d.reason.removeprefix("shed:")
        sources = next((r["sources"] for r in view.reasons if r["name"] == name), [])
        await self._emit(GpuPoolEventV1(
            event="queued", lease_id=d.lease_id, holder=row.get("holder"), work_class=row.get("work_class"),
            priority=row.get("priority"), turn_correlation_id=row.get("turn_correlation_id"),
            attempt=row.get("attempt"), reason=d.reason,
            detail={"shed": True, "shed_reason": name, "sources": [s["source_id"] for s in sources]}))
        self._shed_reported.add(key)   # only once sent: a failed publish is retried next tick

    async def _report_serialized(self, d: Serialized, row: dict | None) -> None:
        """serialize_with (stage 5 Z1): a ``queued`` event naming the blocking role, once per lease
        and blocker while it lasts. Not a lease transition: the row stays queued."""
        key = (d.lease_id, d.reason)
        if key in self._serialized_reported or row is None:
            return
        await self._emit(GpuPoolEventV1(
            event="queued", lease_id=d.lease_id, holder=row.get("holder"), work_class=row.get("work_class"),
            priority=row.get("priority"), role=d.role, cards=list(self.cfg.roles[d.role].cards),
            turn_correlation_id=row.get("turn_correlation_id"), attempt=row.get("attempt"),
            reason=d.reason, detail={"serialized": True}))
        self._serialized_reported.add(key)   # only once sent: a failed publish is retried next tick

    @staticmethod
    def _swap_key(role: str, action: str, reason: str, wanted: str) -> tuple:
        """One reporting episode: seat + action + the reason reported + the scheduler's own reason.
        Never the guard's state text or anything else that can change tick to tick (a key that
        flips re-fires every tick)."""
        return (role, action, reason, wanted)

    async def _swap(self, d: SwapLoad | SwapUnload | SwapBlocked) -> tuple:
        """A seat with a launch block is actuated. While actuation is paused, for a seat with no launch
        block, and for every blocked load, the decision is reported as ``swap_requested {actuated:
        false}`` instead -- once per episode (same seat, action and reason); a new reason, or the block
        clearing and recurring, is a new episode. Returns the episode key the caller must keep alive
        for this tick."""
        action = "unload" if isinstance(d, SwapUnload) else "load"
        if d.role in self._reconciling and not isinstance(d, SwapBlocked):
            # the actuator has not said yet what the card holds: decide again next tick
            return self._swap_key(d.role, action, "reconciling", d.reason)
        paused = self.paused is not None and isinstance(d, (SwapLoad, SwapUnload))
        if isinstance(d, (SwapLoad, SwapUnload)) and d.role in self.actuated and not paused:
            now = self.now()
            if not any(c.cooldown_until and c.cooldown_until > now for c in self._seat_cards(d.role)):
                await self._begin_actuation(d)
                return self._swap_key(d.role, action, "actuated", d.reason)
            # Backoff after a refused/unanswered action. The scheduler already reports blocked
            # LOADS; an unload has no scheduler-side cooldown, and without this a dead actuator
            # would get a fresh unload every actuate_ack_sec.
            d = SwapBlocked(d.role, "cooldown", action)
        detail: dict[str, Any] = {"action": action, "actuated": False, "mode": self.mode}
        reason = d.reason
        if isinstance(d, SwapBlocked):
            detail.update(blocked=True, guard_state=d.detail)
        elif paused:
            detail.update(paused=True, wanted=d.reason)
            reason = "actuation_paused"
        elif d.role not in self.actuated:
            reason = f"not_actuatable:{d.role}"
            detail.update(wanted=d.reason)
        key = self._swap_key(d.role, action, reason, d.reason)
        if key in self._swap_requested:
            return key
        await self._emit(GpuPoolEventV1(
            event="swap_requested", role=d.role, cards=list(self.cfg.roles[d.role].cards), reason=reason,
            detail=detail))
        # After the emit: an emit that raises is retried next tick (publish errors are swallowed inside).
        self._swap_requested.add(key)
        return key

    # --- actuation engine -----------------------------------------------------------------
    def _seat_cards(self, seat: str) -> list[CardLive]:
        return [self.cards[c] for c in self.cfg.roles[seat].cards]

    def _action_timeout(self, seat: str) -> float:
        """The whole action: a load drains+stops what it evicts then starts the seat; an unload
        stops the seat then starts the residents. Budget every launch it touches.

        agent-gpu2 after the stage-5.3 cutover: 900 (seat) + 600 (diffusion) = 1500 s to the first
        `status`, faulted as stuck at MAX_ACTION_TIMEOUTS x that = 3000 s. The controller's realistic
        worst case -- a failed load: two `docker ps` reads (<= 30 s each) + diffusion drain
        (controller GPU_LANE_DRAIN_TIMEOUT_SEC, 300) + seat ready wait (900) + diffusion's restore ready wait (600) +
        four quick `docker stop`/`up` calls -- is ~1900 s: past the first deadline (the pool keeps
        polling while `status` says in_flight) and inside the ceiling. NOT inside it: docker calls
        that each run near their own GPU_LANE_COMMAND_TIMEOUT_SEC (900 s, four of them on a failed
        load, ~4100 s in all). That is a wedged actuator, and faulting the card is this ceiling's job;
        a late result is then ignored and an operator `clear_fault` reconciles. Both bounds pinned by tests/test_stage5_3_cutover_e2e.py."""
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
        # Stage 5.3: a load names the seat's model -- launch.profiles[0], or None (compose default)
        # when the role lists none. An unload stops the seat and restarts residents on their own
        # compose defaults, so it never carries one.
        profile = self.cfg.load_profile(seat) if action == "load" else None
        msg = GpuActuateV1(
            action_id=f"{seat}:{action}:g{generation}:{uuid.uuid4().hex[:8]}", generation=generation,
            actuator=spec.launch.actuator, role=seat, action=action, cards=list(spec.cards), profile=profile,
            launch_digest=launch_digest(self.cfg, seat), deadline_at=now + timedelta(seconds=timeout),
            reason=d.reason)
        act = {"action_id": msg.action_id, "role": seat, "action": action, "generation": generation,
               "profile": profile,
               "reason": d.reason, "sent_at": now.isoformat(), "deadline_at": msg.deadline_at.isoformat(),
               "acked_at": None, "phase": None, "outcome": None, "status_action_id": None,
               "status_sent_at": None, "extensions": 0}
        for c in cards:
            c.swap_state = "loading" if action == "load" else "unloading"
            c.swap_role = seat
            c.swap_generation = generation
            c.swap_action = act
        recent = self._recent_actions.setdefault(seat, [])
        recent.append(msg.action_id)
        del recent[:-RECENT_ACTIONS]
        # Persisted BEFORE the request leaves: a crash after the send still finds the action and
        # reconciles with `status` instead of sending a second transition.
        await self._save_cards(cards, "actuate")
        logger.info("gpu_pool_actuate_send seat=%s action=%s action_id=%s generation=%s profile=%s reason=%s",
                    seat, action, msg.action_id, generation, profile, d.reason)
        await self._emit(GpuPoolEventV1(event="swap_started", role=seat, cards=list(spec.cards), reason=d.reason,
                                        detail={"action": action, "action_id": msg.action_id,
                                                "generation": generation, "profile": profile, "actuated": True}))
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
                        "generation": act.get("generation"), "profile": act.get("profile"),
                        "swap_state": state, "actuated": True, **(detail or {})}))

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
            rec = self._reconciling.get(seat)
            if res.action == "status" and rec is not None and res.action_id == rec["action_id"]:
                self._reconciling.pop(seat, None)
                await self._on_reconcile(seat, res, rec)
                return
            if res.action == "status":
                if not pending or res.action_id != act.get("status_action_id"):
                    # Also the normal end of a reconcile: the actuator re-publishes the last result it
                    # recorded under ITS action_id first, which settled the card before this arrived.
                    logger.info("gpu_pool_actuate_status_stale seat=%s action_id=%s", seat, res.action_id)
                    return
                if res.status in ("accepted", "progress"):
                    return
                if res.status == "refused":
                    self._controller_refused(seat, res.reason)
                elif res.status == "succeeded":
                    self._controller_answered(seat, via_status=True)
                if res.status != "succeeded":
                    await self._finish(seat, state="fault", loaded=None, outcome=f"status_{res.status}",
                                       event="swap_failed", reason=f"status_{res.status}:{res.reason}")
                    return
                # Any status answer proves the actuator is reachable; without acked_at the ack
                # timeout would call a live, running action "actuator_unreachable" next tick.
                acked = act.get("acked_at") or now.isoformat()
                running = res.in_flight
                if running is None:   # an actuator that does not say: a reported phase means mid-action
                    running = res.phase is not None
                if running:
                    self._set_action(seat, status_action_id=None, phase=res.phase, in_flight=True, acked_at=acked,
                                     deadline_at=(now + timedelta(seconds=STATUS_POLL_SEC)).isoformat())
                    await self._save_cards(cards, "actuate_status")
                    logger.info("gpu_pool_actuate_still_running seat=%s action_id=%s phase=%s",
                                seat, act.get("action_id"), res.phase)
                    return
                observed = dict(res.observed)
                if res.in_flight is None:
                    # The actuator cannot say whether our action runs: between `accepted` and its first
                    # `progress` it reports no phase although it is working. Ask again (a running
                    # action reports a phase within one poll) before believing the containers.
                    n = int(act.get("extensions") or 0) + 1
                    if n <= MAX_STATUS_EXTENSIONS:
                        self._set_action(seat, status_action_id=None, extensions=n, acked_at=acked,
                                         deadline_at=(now + timedelta(seconds=STATUS_POLL_SEC)).isoformat())
                        await self._save_cards(cards, "actuate_status")
                        return
                # Nothing running there and our action's own result never came: the card is what the
                # containers say it is (or a fault when they say something half-done).
                await self._adopt(seat, observed, why="status")
                return

            if not pending and rec is not None:
                # The actuator re-publishes its last recorded result before answering a reconcile
                # `status`; that row's `observed` is as old as the action. The fresh answer decides.
                logger.info("gpu_pool_actuate_result_during_reconcile seat=%s action_id=%s", seat, res.action_id)
                return
            if not pending or res.action_id != act.get("action_id"):
                # A result for an action we already gave up on (unanswered in time, or superseded by
                # a refused retry). If this process issued it and the card is idle, believe the
                # containers it reports.
                terminal = res.status in ("succeeded", "failed")
                ours = res.action_id == act.get("action_id") or res.action_id in self._recent_actions.get(seat, [])
                if terminal and ours and cards[0].swap_state == "idle" \
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

            # Our own action's answer: anything but a refusal means the controller read its config
            # (it admits, i.e. resolves against the config, before it says accepted).
            if res.status == "refused":
                self._controller_refused(seat, res.reason)
            else:
                self._controller_answered(seat)
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

    # --- controller health (orion/gpu_pool/controller_health.py) ---------------------------------------
    def _controller_host(self, seat: str) -> str:
        launch = self.cfg.roles[seat].launch
        return launch.actuator if launch else "unknown"

    def _controller_refused(self, seat: str, reason: str | None) -> None:
        t = self.controller_health.on_refused(seat, reason, host=self._controller_host(seat), now=self.now())
        if t is None:
            return
        view = t.view()
        logger.error("gpu_pool_controller_degraded seat=%s host=%s kind=%s reason=%s refusals=%s first_seen=%s -- %s",
                     seat, t.host, t.kind, t.reason, t.count, view["first_seen"], view["advice"])
        self._fire_controller_alert(seat, "degraded", view)

    def _controller_answered(self, seat: str, *, via_status: bool = False) -> None:
        t = self.controller_health.on_answered(seat, via_status=via_status)
        if t is None:
            return
        view = {**t.view(), "degraded": False, "recovered_at": self.now().isoformat()}
        logger.warning("gpu_pool_controller_recovered seat=%s host=%s kind=%s refusals=%s degraded_since=%s",
                       seat, t.host, t.kind, t.count, view["degraded_since"])
        self._fire_controller_alert(seat, "recovered", view)

    def _fire_controller_alert(self, seat: str, state: str, view: dict[str, Any]) -> None:
        """Fire-and-forget: the alert does HTTP and must never hold the lease lock or fail a result."""
        if self.controller_alert is None:
            return

        async def send() -> None:
            try:
                await self.controller_alert(seat, state, view)
            except Exception:  # noqa: BLE001
                logger.exception("gpu_pool_controller_alert_failed seat=%s state=%s", seat, state)

        task = asyncio.create_task(send())
        self._alert_tasks.add(task)
        task.add_done_callback(self._alert_tasks.discard)

    async def drain_alerts(self, timeout: float = 5.0) -> None:
        """Shutdown: let an in-flight alert POST finish (bounded), then cancel what is left."""
        tasks = list(self._alert_tasks)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def _card_actuation(self, c: CardLive) -> dict[str, Any] | None:
        """The card's action record, plus ``controller_degraded`` (not persisted) for a degraded seat on
        it: the state payload's card ``actuation`` is a free dict, so old consumers just carry it."""
        bad = {s: t.view() for s, t in self.controller_health.degraded().items()
               if c.card in self.cfg.roles[s].cards}
        if not bad:
            return c.swap_action
        return {**(c.swap_action or {}), "controller_degraded": bad}

    async def _check_actuation(self) -> None:
        """Timeouts (spec "Timeouts"): no accepted within actuate_ack_sec -> actuator_unreachable;
        no terminal result by deadline_at -> ask `status`; no answer to that -> fault."""
        now = self.now()
        ack = self.cfg.defaults.actuate_ack_sec
        for seat, rec in list(self._reconciling.items()):
            if (now - rec["sent_at"]).total_seconds() >= max(ack, STATUS_REPLY_SEC):
                # Nothing was in flight, so nothing to fault: keep the persisted state, say so, and
                # let the next real action meet the actuator (unreachable -> the usual cooldown).
                self._reconciling.pop(seat, None)
                logger.warning("gpu_pool_reconcile_unanswered seat=%s action_id=%s why=%s -- keeping the "
                               "persisted card state", seat, rec["action_id"], rec["why"])
        for seat, act in self._pending_actions().items():
            sent0 = _ts(act.get("sent_at"))
            if sent0 and (now - sent0).total_seconds() >= MAX_ACTION_TIMEOUTS * self._action_timeout(seat):
                await self._finish(seat, state="fault", loaded=None, outcome="stuck", event="swap_failed",
                                   reason="actuator_stuck", detail={"phase": act.get("phase")})
                continue
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

    # --- stage 5.7: adoption and the emergency stop ---------------------------------------
    async def _reconcile_idle_seats(self, why: str) -> None:
        """enforce mode: ask the actuator (one read-only ``status`` per seat) what each idle actuated
        seat's cards really hold, and adopt the answer. This is how a seat loaded or unloaded outside
        the pool (by hand, while paused, or before this process) is adopted, never reloaded. observe
        mode keeps the older liveness shortcut instead (``_observe_swap_seats``)."""
        if self.mode != "enforce":
            return
        for seat in sorted(self.actuated):
            cards = self._seat_cards(seat)
            if any(c.swap_state != "idle" for c in cards) or seat in self._reconciling:
                continue   # in flight: the pending-action reconcile owns it; fault: clear_fault does
            spec = self.cfg.roles[seat]
            now = self.now()
            rec = {"action_id": f"{seat}:reconcile:{uuid.uuid4().hex[:8]}", "sent_at": now, "why": why}
            self._reconciling[seat] = rec
            msg = GpuActuateV1(
                action_id=rec["action_id"], generation=max(1, max(c.swap_generation for c in cards)),
                actuator=spec.launch.actuator, role=seat, action="status", cards=list(spec.cards), profile=None,
                launch_digest=launch_digest(self.cfg, seat),
                deadline_at=now + timedelta(seconds=self.cfg.defaults.actuate_ack_sec), reason=f"reconcile:{why}")
            logger.info("gpu_pool_reconcile_send seat=%s action_id=%s why=%s", seat, rec["action_id"], why)
            await self._publish(GPU_POOL_ACTUATE_REQUEST_CHANNEL, GPU_ACTUATE_KIND, msg.model_dump(mode="json"), None)

    async def _on_reconcile(self, seat: str, res: GpuActuateResultV1, rec: dict[str, Any]) -> None:
        why = rec["why"]
        cards = self._seat_cards(seat)
        if any(c.swap_state != "idle" for c in cards):
            logger.info("gpu_pool_reconcile_superseded seat=%s swap_state=%s", seat, cards[0].swap_state)
            return
        if res.status in ("accepted", "progress"):
            self._reconciling[seat] = rec   # not the answer yet
            return
        if res.status == "refused":
            self._controller_refused(seat, res.reason)
        elif res.status == "succeeded":
            self._controller_answered(seat, via_status=True)
        if res.status != "succeeded":
            logger.warning("gpu_pool_reconcile_refused seat=%s status=%s reason=%s -- keeping the persisted "
                           "card state", seat, res.status, res.reason)
            return
        observed = dict(res.observed)
        if res.in_flight is None:
            # An actuator that cannot say whether something runs: believe neither way, keep the card.
            logger.warning("gpu_pool_reconcile_unknown_in_flight seat=%s -- keeping the persisted card state", seat)
            return
        loaded = None if res.in_flight else self._observed_outcome(seat, observed)
        if loaded is not None and loaded == all(seat in c.swapped_in for c in cards):
            # Nothing changed: the card keeps its last load/unload record (the Hub shows that, not this).
            logger.info("gpu_pool_reconcile_agrees seat=%s loaded=%s why=%s", seat, loaded, why)
            return
        # The card changes (adopted or faulted): it gets this reconcile as its action record, so its
        # outcome is never read as the previous load/unload's.
        gen = max(c.swap_generation for c in cards)
        for c in cards:
            c.swap_action = {"action_id": rec["action_id"], "role": seat, "action": "status", "generation": gen,
                             "profile": None, "reason": f"reconcile:{why}", "sent_at": rec["sent_at"].isoformat(),
                             "acked_at": self.now().isoformat(), "phase": None, "outcome": None,
                             "status_action_id": None, "status_sent_at": None, "extensions": 0}
        if res.in_flight:
            # The actuator is running an action this pool never sent (or no longer knows about).
            await self._finish(seat, state="fault", loaded=None, outcome="reconcile_foreign_action",
                               event="swap_failed", reason=f"reconcile_foreign_action:{why}",
                               detail={"observed": observed, "phase": res.phase})
        elif loaded is None:
            await self._finish(seat, state="fault", loaded=None, outcome="reconcile_ambiguous", event="swap_failed",
                               reason=f"reconcile_ambiguous:{why}", detail={"observed": observed})
        else:
            await self._finish(seat, state="idle", loaded=loaded, outcome="adopted", event="swapped",
                               reason=f"adopted:{why}", detail={"observed": observed, "loaded": loaded})

    async def _set_paused(self, pause: bool, actor: str) -> GpuPoolControlReplyV1:
        """The emergency stop. Persisted on every gpu_pool_cards row, so a restart stays paused. An
        action already in flight is NOT stopped (the actuator owns it; stop circe's
        orion-gpu-lane-controller for that); the pool keeps following its result."""
        async with self._locked("control"):
            if pause == (self.paused is not None):
                return GpuPoolControlReplyV1(ok=True, reason="already_paused" if pause else "not_paused",
                                             detail=self._paused_detail())
            now = self.now()
            try:
                # One statement over EVERY card row (also rows of cards since removed from the YAML), and
                # before memory: a failed write must not leave the pool running while the rows say paused.
                await self.store.set_actuation_paused(now if pause else None, actor if pause else None, now, actor)
            except Exception as exc:  # noqa: BLE001
                logger.exception("gpu_pool_actuation_pause_not_persisted pause=%s actor=%s", pause, actor)
                return GpuPoolControlReplyV1(ok=False, reason=f"not_persisted:{type(exc).__name__}",
                                             detail=self._paused_detail())
            self.paused = {"since": now, "by": actor} if pause else None
            in_flight = sorted(self._pending_actions())
            log = logger.warning if pause else logger.info
            log("gpu_pool_actuation_%s actor=%s in_flight=%s", "paused" if pause else "resumed", actor, in_flight or "-")
            await self._emit(GpuPoolEventV1(event="actuation_paused" if pause else "actuation_resumed",
                                            holder=actor, cards=list(self.cfg.cards),
                                            detail={"in_flight": in_flight, "actuated": sorted(self.actuated)}))
            if not pause:
                # Whatever an operator did by hand while paused: ask, adopt, then decide.
                await self._reconcile_idle_seats("resume")
            await self._schedule_and_apply()
            await self.publish_state()
        return GpuPoolControlReplyV1(ok=True, reason="paused" if pause else "resumed",
                                     detail={**self._paused_detail(), "in_flight": in_flight})

    def hold_limits(self) -> dict[str, dict[str, Any]]:
        """Stage 7.3: each role configured above one hold (or with a one-off reserve), as the
        scheduler applies it right now: configured max_holds, discovered slots, the limit in force,
        and why it is lower (None when it is not). /health shows it; 7.4 puts holds/max_holds in
        pool state and Hub."""
        out: dict[str, dict[str, Any]] = {}
        for role, spec in self.cfg.roles.items():
            if spec.max_holds <= 1 and not spec.reserve_one_off_slots:
                continue
            live = self.roles.get(role)
            cap, reason = hold_cap(self.cfg, role, live)
            out[role] = {"max_holds": spec.max_holds, "reserve_one_off_slots": spec.reserve_one_off_slots,
                         "slots": live.slots if live else 0, "effective": cap, "reason": reason}
        return out

    def _report_hold_caps(self) -> None:
        """Edge-triggered: say once when a role's hold limit is clamped below its max_holds (fewer
        discovered slots than configured, or the one-off reserve), and once when it clears. An
        unloaded seat (no slots) is not news: it takes no holds of any kind."""
        for role, row in self.hold_limits().items():
            reason = row["reason"]
            if reason == "no_slots" or self._hold_cap_reported.get(role, "unset") == reason:
                continue      # an unloaded seat changes nothing: keep the last state said
            self._hold_cap_reported[role] = reason
            if reason:
                logger.warning("gpu_pool_max_holds_clamped role=%s max_holds=%s effective=%s slots=%s reason=%s",
                               role, row["max_holds"], row["effective"], row["slots"], reason)
            else:
                logger.info("gpu_pool_max_holds_in_force role=%s max_holds=%s slots=%s",
                            role, row["max_holds"], row["slots"])

    def _paused_detail(self) -> dict[str, Any]:
        if self.paused is None:
            return {"paused": False}
        return {"paused": True, "since": self.paused["since"].isoformat(), "by": self.paused["by"]}

    async def _clear_fault(self, ctl: GpuPoolControlV1) -> GpuPoolControlReplyV1:
        """Operator: take a card out of fault. A seat still in the YAML is reconciled with the
        actuator (`status` -> adopt what it reports; unanswered -> fault again, visibly). A seat the
        YAML no longer knows, or a card with nothing to ask about, settles from discovery (seat
        worker alive -> loaded, else not) with a cooldown before any new load."""
        async with self._locked("control"):
            card = self.cards.get(ctl.card or "")
            if card is None:
                return GpuPoolControlReplyV1(ok=False, reason="unknown_card")
            if card.swap_state != "fault":
                return GpuPoolControlReplyV1(ok=False, reason=f"not_faulted:{card.swap_state}")
            seat = card.swap_role
            spec = self.cfg.roles.get(seat or "")
            if spec is not None and spec.swap is not None and spec.launch is not None:
                for c in self._seat_cards(seat):   # whatever it settles to, no new load right away
                    c.cooldown_until = self.now() + timedelta(seconds=self.cfg.defaults.swap_cooldown_sec)
                for c in self._seat_cards(seat):
                    c.swap_state = "loading" if not all(seat in x.swapped_in for x in self._seat_cards(seat)) \
                        else "unloading"
                act = self._set_action(seat, status_action_id=None, extensions=MAX_STATUS_EXTENSIONS,
                                       sent_at=self.now().isoformat(), acked_at=self.now().isoformat(),
                                       outcome=None, reason=f"operator_clear:{ctl.actor}")
                await self._send_status(seat, act, reason=f"operator_clear:{ctl.actor}")
                return GpuPoolControlReplyV1(ok=True, reason="reconciling", detail={"card": card.card, "seat": seat})
            loaded_roles = {r for r in card.swapped_in if r in self.cfg.roles and self._seat_alive(r)}
            card.swapped_in = loaded_roles
            card.swap_state, card.swap_role = "idle", None
            card.cooldown_until = self.now() + timedelta(seconds=self.cfg.defaults.swap_cooldown_sec)
            await self._save_cards([card], f"operator:{ctl.actor}")
            await self._emit(GpuPoolEventV1(event="swapped", cards=[card.card], holder=ctl.actor,
                                            reason="fault_cleared:operator", detail={"swap_role": seat}))
            return GpuPoolControlReplyV1(ok=True, reason="cleared", detail={"card": card.card})

    async def _save_seen_ctx(self) -> None:
        """Persist each role's last-seen context on its card, only when it changed (rare): the load
        decision for an unloaded seat needs it after a pool restart too."""
        changed = {r for r, n in self._ctx_seen.items() if self._ctx_saved.get(r) != n}
        if not changed:
            return
        for card in {c for r in changed for c in self.cfg.roles[r].cards}:
            sizes = {r: n for r, n in self._ctx_seen.items() if card in self.cfg.roles[r].cards}
            await self.store.upsert_card({"card": card, "seen_ctx": sizes, "updated_at": self.now(),
                                          "updated_by": "discovery"})
        self._ctx_saved.update({r: self._ctx_seen[r] for r in changed})

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
            reason=row.get("reason"),
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
            served_by=f"{self.cfg.role_host(role)}-worker-{role}")

    async def _reply_for(self, row: dict) -> GpuLeaseReplyV1:
        status = row["status"]
        if status == "granted":
            return GpuLeaseReplyV1(status="granted", lease_id=row["lease_id"],
                                   grant=self.grant_for(row["role"], row["lease_id"], row["generation"]))
        if status == "recalling":
            # reason = why it is recalled; a durable run waits out an urgent_preempt (re-queued in place).
            return GpuLeaseReplyV1(status="recall", lease_id=row["lease_id"], recall_by=row["recall_by"],
                                   reason=row.get("reason"),
                                   grant=self.grant_for(row["role"], row["lease_id"], row["generation"]))
        if status in ("queued", "retry_wait"):
            return GpuLeaseReplyV1(status="queued", lease_id=row["lease_id"], position=await self._position(row),
                                   reason=row.get("reason"))
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
            host=self.cfg.host.name,
            cards=[GpuCardStateV1(card=c.card, vram_gb=self.cfg.cards[c.card].vram_gb,
                                  index=self.cfg.cards[c.card].index,
                                  host=self.cfg.card_host(c.card),
                                  lendable=self.cfg.cards[c.card].lendable, lent=c.lent,
                                  swapped_in=sorted(c.swapped_in), swap_state=c.swap_state,
                                  cooldown_until=c.cooldown_until, swap_role=c.swap_role,
                                  residency_until=c.residency_until, loaded_at=c.loaded_at,
                                  actuated_roles=sorted(r for r in self.actuated
                                                        if c.card in self.cfg.roles[r].cards),
                                  actuation=self._card_actuation(c)) for c in self.cards.values()],
            roles=self.discovered, unclaimed_servers=self.unclaimed,
            leases=[GpuLeaseRowV1(
                lease_id=r["lease_id"], request_id=r["request_id"], holder=r["holder"],
                work_class=r["work_class"], priority=r["priority"], kind=r["kind"], status=r["status"],
                role=r.get("role"), attempt=r.get("attempt", 1), created_at=r["created_at"],
                granted_at=r.get("granted_at"), recall_by=r.get("recall_by"),
                turn_correlation_id=r.get("turn_correlation_id"), generation=int(r.get("generation") or 0),
                hold_lease_id=r.get("hold_lease_id")) for r in rows] if include_leases else [],
            queue_depth=queue, backlog_depth=backlog, swap_guards=dict(self.guard_states),
            actuation_paused=self._paused_detail() if self.paused is not None else None,
            shed={**self.shed_board.view(self.now(), self.shed_enabled).as_dict(),
                  "orion_self_shed": self.orion_shed.health()},
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
        """Pool events travel on their OWN envelope correlation id (derived from ``event_id``), never
        the turn's. bus-mirror turns every shared envelope correlation_id into CAUSALLY_FOLLOWED_BY
        edges, so publishing on the turn's id put orion-gpu-pool inside turn causal chains (36 live
        pool edges on 2026-09-29, feeding bus_synaptic_prediction_error). Spec 2026-09-24-gpu-pool-design.md,
        "Transport-metric and reader impacts" item 1. No consumer reads the envelope correlation_id;
        the turn stays joinable through the payload's ``turn_correlation_id``."""
        if self.bus is None:
            return
        await self._publish(GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND, event.model_dump(mode="json"),
                            event_envelope_correlation(event))
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
                source_name=self.service_name, correlation_id=event_envelope_correlation(event))
        except Exception:  # noqa: BLE001
            logger.warning("gpu_pool_grammar_failed event=%s", event.event, exc_info=True)
