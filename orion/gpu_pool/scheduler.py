"""The pool's only decision-maker: a pure function from (config, live roles, cards, leases, now)
to a list of decisions. No I/O, no clock, no randomness, so every rule in the spec is a unit
test and the eval can replay a day of traffic through it.

Rules implemented (numbers match docs/superpowers/specs/2026-09-24-gpu-pool-design.md):
  2  context and capability requirements are checked against the discovered model
  3  home first
  4  borrowing only along the YAML's class -> roles lists
  5  owners win on their own role: borrowers are never granted while an owner waits, and are
     recalled (grace, then abort) when an owner is starved
  6  queue order: priority, then oldest
  7  swap seats: load only when the evicted residents are idle; reclaimed by the residents'
     owners; cooldown; idle unload
  8  operator-only multi-card seats drain everything they evict
  9  lendable cards: non-owners only while lent; unlending recalls borrowers
  10 on_unavailable: wait / backlog / fail

Stage 4.3 (docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md):
  H1 a hold (kind="hold") is placed like any lease and reserves one slot; a role takes at most
     ``hold_cap`` non-urgent holds at once (stage 7.3: roles.<r>.max_holds, default 1, clamped to the
     discovered slots, less roles.<r>.reserve_one_off_slots but never below one hold)
  H2 a child (hold_lease_id set) runs in its hold's slot: only on the hold's role, ahead of the
     role's queue, never behind its own run, never taking a second slot for the pair
  H3 interleave: while a hold has no child in flight, a lease of STRICTLY higher priority may use
     that slot (gaps are shared, Juniper 2026-09-25); equal/lower priority and other holds may not.
     Stage 7.3 gap pinning: a call using a gap is charged to ONE idle hold (rebuilt every tick from
     the active leases, ``_Ctx.charged``): an idle hold whose run has no call waiting first, then the
     most recently granted. Only the charged run waits for it; another idle run's next call gets its
     own slot at once. Without this one interloper on a 2-hold role stalled both runs
  H4 hold recall uses defaults.hold_clawback_grace_sec; a swap seat with max_hold_sec drains once
     it has been loaded that long, then unloads (today's DURABLE_RUNS_ELASTIC_MAX_BORROW_SEC)
  S1 a seat load is blocked -- reported as SwapBlocked, never silent -- by min residency after an
     unload, cooldown after a failed load, or a failing guard (thermal)
  S2 a card in "fault" grants nothing on any of its roles; a card mid-swap grants nothing on the
     seat or the roles it evicts

Urgent (docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md):
  U1 a waiting urgent lease that got no slot pauses one granted background (then system)
     durable-run hold on a role it could use: recall urgent_preempt with urgent_preempt_grace_sec;
     most recently granted first; never interactive, urgent, operator or request leases, and never
     the owner of a role the urgent lease only borrows (re-queued, the owner would win it back).
     A pause under way (recalling urgent_preempt, or re-queued urgent_preempt with its last call
     still on the slot) on a slot the lease could take serves it first. Only urgent leases the
     cap has room for are owed a pause or count as waiting owners.
     An urgent owner reclaiming its own role pauses that role's pausable borrowing hold whatever
     its priority (system as well as background, even if a background hold elsewhere could be
     paused instead): that is its pause (urgent_preempt, short grace), not owner_waiting with the
     hold grace
  U2 the paused hold's abort re-queues it in place without spending an attempt (lease_graph); so does
     the abort of any other recalled retryable hold (H4 max_hold, owner reclaim, unlend, drain)
  U3 urgent holds are exempt from H1 (bounded by slots and urgent_max_concurrent) and skip a
     swap seat's after_wait_sec when no pause serves them (guards still apply). On a role it
     borrows, an urgent hold's gaps stay open to that role's owners (rule 5)
  urgent_max_concurrent caps active urgent leases; 0 is the rollback: urgent is background
  U4 shed (docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md):
     ``shed`` maps a priority to the shed reason blocking it (orion/gpu_pool/shed.py decides which;
     only background/system are ever passed). A queued, backlogged or retrying lease of a shed
     priority gets no NEW grant and is reported as Shed(reason="shed:<name>"); it is not demand
     anywhere (no owner-waiting recall, no swap load, no seat drain, no wait/backlog/fail verdict)
     and keeps its place in line and its deadline. Running work is untouched: nothing is recalled,
     and a granted hold's own calls (children) are still granted. Decided on the lease's ORIGINAL
     priority, so the urgent rollback never makes urgent work sheddable.
     One-shot refusal (D3, docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md):
     a ONE-SHOT request -- kind "request", no hold_lease_id, status "queued" at the start of the
     tick, and not a lease someone comes back for (its class is not on_unavailable "backlog" with
     retryable set; a non-retryable backlog lease already behaves like "wait", rule 10) -- is
     refused at once with Unavailable(reason="shed:<name>") instead of waiting out its deadline,
     so the caller's fallback runs now. Durable-run holds, backlog leases and retry_wait leases
     keep the Shed (wait) behavior above; a retry_wait lease is Requeue'd this tick and is not
     also refused, so one lease never gets two transitions in one tick

Stage 5 (docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md):
  Z1 serialize_with: nothing is placed on a role while a lease (request, hold or child) is active on
     a role it serializes with, in either direction. No recall and no preemption across the pair:
     the waiting lease keeps its place in queue order (priority, then age) and its own deadline --
     a lease blocked only by the mutex reserves the partner role, so later leases are not granted
     there and the partner drains. A lease a free role would otherwise
     take is reported as Serialized(reason="serialized:<role>"), never silent
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Collection, Iterable, Mapping, Union

from orion.gpu_pool.config import PoolConfig
from orion.schemas.gpu_pool import URGENT_PREEMPT as PREEMPT

ACTIVE = ("granted", "recalling")
URGENT = "urgent"
PREEMPTIBLE = ("background", "system")


@dataclass(frozen=True)
class RoleLive:
    """What discovery knows about a role right now."""

    role: str
    healthy: bool                 # confirmed announcement + /props agree, or service /health ok
    slots: int = 0
    ctx_per_slot: int | None = None
    vision: bool | None = None


@dataclass
class CardLive:
    card: str
    lent: bool = False
    swapped_in: set[str] = field(default_factory=set)
    swap_state: str = "idle"      # idle | loading | unloading | fault
    cooldown_until: datetime | None = None   # set after a failed/refused load; blocks reload
    last_active_at: datetime | None = None   # last time a lease held a swap seat on this card
    swap_role: str | None = None             # the seat a loading/unloading/fault state is about
    residency_until: datetime | None = None  # after an unload, the evicted residents stay until this
    loaded_at: datetime | None = None        # when the seat on this card was loaded (max_hold_sec)
    # Runtime bookkeeping for the actuation engine; the scheduler reads neither.
    swap_generation: int = 0
    swap_action: dict | None = None


@dataclass(frozen=True)
class LeaseView:
    lease_id: str
    work_class: str
    priority: str
    status: str
    created_at: datetime
    role: str | None = None
    min_ctx_tokens: int = 0
    needs_vision: bool = False
    deadline_at: datetime | None = None
    recall_by: datetime | None = None
    not_before: datetime | None = None
    queued_since: datetime | None = None
    granted_at: datetime | None = None
    expires_at: datetime | None = None
    operator: bool = False
    retryable: bool = False   # someone will use a re-grant; otherwise "backlog" behaves like "wait"
    kind: str = "request"     # request | hold
    hold_lease_id: str | None = None   # set on a child: the hold whose slot it runs in
    reason: str | None = None          # the row's last transition reason (the recall reason while recalling)


@dataclass(frozen=True)
class Grant:
    lease_id: str
    role: str


@dataclass(frozen=True)
class Recall:
    lease_id: str
    recall_by: datetime
    reason: str


@dataclass(frozen=True)
class Abort:
    lease_id: str
    reason: str = "recall_grace_exceeded"


@dataclass(frozen=True)
class Expire:
    lease_id: str
    reason: str = "heartbeat_lost"


@dataclass(frozen=True)
class Unavailable:
    lease_id: str
    reason: str


@dataclass(frozen=True)
class Backlog:
    lease_id: str
    reason: str


@dataclass(frozen=True)
class Requeue:
    lease_id: str
    reason: str


@dataclass(frozen=True)
class DeadLetter:
    lease_id: str
    reason: str


@dataclass(frozen=True)
class SwapLoad:
    role: str
    reason: str


@dataclass(frozen=True)
class SwapUnload:
    role: str
    reason: str


@dataclass(frozen=True)
class SwapBlocked:
    """A load the demand justifies but a precondition refuses: min_residency, cooldown or
    guard:<name>. Reported (edge-triggered by the runtime), never actuated."""
    role: str
    reason: str
    detail: str | None = None


@dataclass(frozen=True)
class Serialized:
    """A queued lease a role could otherwise take right now, held back by ``serialize_with``
    (stage 5): ``reason`` is ``serialized:<role>`` naming the role whose active lease blocks it.
    Reported (edge-triggered by the runtime), never a lease transition: the lease stays queued."""
    lease_id: str
    role: str
    reason: str


@dataclass(frozen=True)
class Shed:
    """A lease a shed reason keeps waiting (U4): ``reason`` is ``shed:<name>``. Reported
    (edge-triggered by the runtime), never a lease transition: the lease stays where it is."""
    lease_id: str
    reason: str


Decision = Union[Grant, Recall, Abort, Expire, Unavailable, Backlog, Requeue, DeadLetter, SwapLoad, SwapUnload,
                 SwapBlocked, Serialized, Shed]


def _one_shot(cfg: PoolConfig, lease: LeaseView) -> bool:
    """D3: a lease nobody comes back for -- refused at once when shed instead of waiting.

    Status "queued" only: a retry_wait lease got Requeue this tick (and is retryable by definition),
    a backlogged one is a backlog lease. "backlog" counts only with ``retryable``: without it rule 10
    already treats the class as "wait" (the gateway's metacog/agent calls are this case)."""
    if lease.kind != "request" or lease.hold_lease_id is not None or lease.status != "queued":
        return False
    return not (cfg.classes[lease.work_class].on_unavailable == "backlog" and lease.retryable)


def hold_cap(cfg: PoolConfig, role: str, live: RoleLive | None) -> tuple[int, str | None]:
    """H1 (stage 7.3): how many non-urgent holds ``role`` may carry now, and why it is below the
    configured ``max_holds`` (None when it is not). Bounded by the DISCOVERED slots: a profile that
    comes up with fewer slots than the YAML expects never gets more holds than it can run. A
    ``reserve_one_off_slots`` reserve is taken from the slots beyond the first hold only, so a 1-slot
    role always runs one. A role with no live slots takes no hold (it is not usable anyway)."""
    spec = cfg.roles[role]
    slots = live.slots if live is not None else 0
    if slots <= 0:
        return 0, "no_slots"
    cap, reason = spec.max_holds, None
    if slots < cap:
        cap, reason = slots, f"max_holds {spec.max_holds} > discovered slots {slots}"
    if spec.reserve_one_off_slots:
        room = max(1, slots - spec.reserve_one_off_slots)
        if room < cap:
            cap, reason = room, f"reserve_one_off_slots {spec.reserve_one_off_slots} of {slots} slots"
    return cap, reason


@dataclass
class _Ctx:
    cfg: PoolConfig
    roles: dict[str, RoleLive]
    cards: dict[str, CardLive]
    now: datetime
    used: dict[str, int]                  # active requests + children, per role
    holds: dict[str, list[LeaseView]]     # active holds, per role (hold_cap non-urgent, + urgent)
    busy_holds: set[str]                  # holds with a child in flight (the child sits in `used`)
    draining: set[str]
    operator_granted: set[str]
    # Stage 7.3 gap pinning: holds whose run has a call waiting this tick (charged last).
    wanting: set[str] = field(default_factory=set)

    def charged(self, role: str) -> set[str]:
        """H3 gap pinning: the idle holds on ``role`` whose gap a one-off call is using right now.

        Rebuilt from active leases on every call, never stored: the slot count says how many idle
        holds are lent out (occupancy beyond the slots), not which -- so each lent gap is charged to
        the idle hold that loses least: one whose run has no call waiting, then the most recently
        granted (the urgent pause's victim order). One borrow charges one hold, so it stalls at most
        one run. At one slot and one hold this is exactly the old rule."""
        live = self.roles.get(role)
        slots = live.slots if live else 0
        idle = [h for h in self.holds.get(role, []) if h.lease_id not in self.busy_holds]
        lent = self.used.get(role, 0) + len(idle) - slots
        if lent <= 0 or not idle:
            return set()
        idle.sort(key=lambda h: (h.lease_id in self.wanting,
                                 -(h.granted_at or h.created_at).timestamp(), h.lease_id))
        return {h.lease_id for h in idle[:lent]}

    def loaded(self, role: str) -> bool:
        spec = self.cfg.roles[role]
        if spec.swap is not None:
            return all(role in self.cards[c].swapped_in for c in spec.cards)
        return not self.evicted(role)

    def evicted(self, role: str) -> bool:
        for other, spec in self.cfg.roles.items():
            if spec.swap is None or other == role:
                continue
            if any(other in self.cards[c].swapped_in for c in spec.cards) and role in self.cfg.evicted_by(other):
                return True
        return False

    def swap_blocked(self, role: str) -> bool:
        """S2: a faulted card grants nothing; a card mid-swap grants nothing on the seat or on
        what it evicts (a mid-swap card with no recorded seat blocks everything on it)."""
        for c in self.cfg.roles[role].cards:
            card = self.cards[c]
            if card.swap_state == "fault":
                return True
            if card.swap_state in ("loading", "unloading"):
                seat = card.swap_role
                if seat is None or seat not in self.cfg.roles or role == seat \
                        or role in self.cfg.evicted_by(seat):
                    return True
        return False

    def usable(self, role: str) -> bool:
        live = self.roles.get(role)
        return bool(live and live.healthy and live.slots > 0 and self.loaded(role)
                    and not self.swap_blocked(role))

    def idle_holds(self, role: str) -> list[LeaseView]:
        """Holds whose slot is empty right now: no child in flight and no one-off in their gap."""
        charged = self.charged(role)
        return [h for h in self.holds.get(role, [])
                if h.lease_id not in self.busy_holds and h.lease_id not in charged]

    def occupancy(self, role: str) -> int:
        """Slots taken, counting a hold with no child in flight as holding its one slot."""
        return self.used.get(role, 0) + len(self.idle_holds(role))

    def free_for(self, lease: LeaseView, role: str) -> int:
        """H1-H3: free slots on ``role`` as ``lease`` sees them."""
        if lease.kind == "hold" and lease.priority != URGENT \
                and len(self.holds.get(role, [])) >= hold_cap(self.cfg, role, self.roles.get(role))[0]:
            return 0  # H1: the role's hold limit (U3: urgent holds stack, bounded by slots)
        live = self.roles.get(role)
        n = (live.slots if live else 0) - self.used.get(role, 0)
        rank = self.cfg.priority_rank
        owns = self.cfg.owns
        for hold in self.idle_holds(role):
            if lease.hold_lease_id == hold.lease_id:
                continue  # a child runs in its own hold's slot
            if lease.kind != "hold" and lease.hold_lease_id is None:
                if rank(lease.priority) < rank(hold.priority):
                    continue  # strictly higher priority may use the gap between the run's calls
                if hold.priority == URGENT and owns(lease.work_class, role) \
                        and not owns(hold.work_class, role):
                    continue  # rule 5: an urgent borrower never outranks the role's owners
            n -= 1
        return n

    def fits(self, lease: LeaseView, role: str) -> bool:
        spec = self.cfg.roles[role]
        if spec.operator_only and not lease.operator:
            return False
        if role in self.draining:
            return False
        if not self.cfg.owns(lease.work_class, role):
            for card in self.cfg.lendable_cards(role):
                if not self.cards[card].lent:
                    return False
        live = self.roles.get(role)
        if live is None:
            return False
        if lease.min_ctx_tokens and spec.kind == "llm":
            if live.ctx_per_slot is None or live.ctx_per_slot < lease.min_ctx_tokens:
                return False
        if lease.needs_vision and not live.vision:
            return False
        return True

    def fits_for_load(self, lease: LeaseView, role: str, seen_ctx: dict[str, int]) -> bool | None:
        """``fits`` for the LOAD decision only: an unloaded seat reports no live context, so its
        expected context is the last one seen there. None = cannot tell (never seen): the caller
        reports that instead of staying silent. Placement still uses live context alone."""
        spec = self.cfg.roles[role]
        live = self.roles.get(role)
        if live is None:
            return False
        if lease.min_ctx_tokens and spec.kind == "llm":
            expected = live.ctx_per_slot or seen_ctx.get(role)
            if expected is None:
                probe = RoleLive(role, live.healthy, live.slots, None, live.vision)
                return None if self._fits_ignoring_ctx(lease, role, probe) else False
            live = RoleLive(role, live.healthy, live.slots, expected, live.vision)
        return self._fits_ignoring_ctx(lease, role, live) and (
            not lease.min_ctx_tokens or spec.kind != "llm" or live.ctx_per_slot >= lease.min_ctx_tokens)

    def _fits_ignoring_ctx(self, lease: LeaseView, role: str, live: RoleLive) -> bool:
        spec = self.cfg.roles[role]
        if spec.operator_only and not lease.operator:
            return False
        if role in self.draining:
            return False
        if not self.cfg.owns(lease.work_class, role):
            for card in self.cfg.lendable_cards(role):
                if not self.cards[card].lent:
                    return False
        if lease.needs_vision and not live.vision:
            return False
        return True

    def serialized_by(self, role: str) -> str | None:
        """The ``serialize_with`` partner (either direction) with an active lease, or None. Reads
        ``used``/``holds``, which this tick's grants update, so two partners are never both granted
        in one tick."""
        for other in self.cfg.serialized_with(role):
            if self.used.get(other) or self.holds.get(other):
                return other
        return None

    def placeable(self, lease: LeaseView, role: str) -> bool:
        return self.usable(role) and self.fits(lease, role) and self.serialized_by(role) is None


def _order(cfg: PoolConfig, leases: Iterable[LeaseView]) -> list[LeaseView]:
    return sorted(leases, key=lambda l: (cfg.priority_rank(l.priority), l.created_at, l.lease_id))


def schedule(
    cfg: PoolConfig,
    roles: dict[str, RoleLive],
    cards: dict[str, CardLive],
    leases: list[LeaseView],
    now: datetime,
    seen_ctx: dict[str, int] | None = None,
    guards: dict[str, str | None] | None = None,
    shed: Mapping[str, str] | None = None,
    frozen: Collection[str] | None = None,
) -> list[Decision]:
    """``seen_ctx``: each role's last-seen per-slot context, kept by the caller across restarts of
    the role. Used ONLY to decide "too big for this class" -- a briefly-down big role must not make
    its class look small. Placement, swaps and serviceability use live ``roles`` alone.

    ``guards``: swap-guard name -> None when clear, else why it fails (a guard the caller could not
    read must be passed as failing, e.g. "unavailable"). A name missing from a dict fails closed.
    ``None`` means the caller evaluates no guards at all (pure tests, the replay eval).

    ``shed``: priority -> shed reason name (U4). None or empty = nothing shed.

    ``frozen``: swap seats whose load/unload cannot happen right now (stage 5.7: the operator paused
    actuation). A swap seat with no ``launch`` block is always frozen (nothing can load it). A frozen
    seat is never drained -- not for an owner reclaim, not at max_hold_sec, and an operator lease on
    it never drains its residents -- because a drain only exists to empty a card for a swap, and a
    swap that cannot happen would leave the card serving nobody. Swap decisions are still returned,
    so the caller can report them."""
    d = cfg.defaults
    out: list[Decision] = []
    # U4, on the ORIGINAL priority (before the urgent rollback below rewrites urgent to background).
    shed_of = {l.lease_id: shed[l.priority] for l in leases
               if shed and l.hold_lease_id is None and l.priority in shed
               and l.status in ("queued", "backlogged", "retry_wait")}
    if d.urgent_max_concurrent <= 0:
        # Rollback switch: urgent behaves exactly like background (no pause, no stacking).
        leases = [replace(l, priority="background") if l.priority == URGENT else l for l in leases]

    active_now = [l for l in leases if l.status in ACTIVE and l.role
                  and not (l.expires_at is not None and l.expires_at <= now)]
    used: dict[str, int] = {}
    holds: dict[str, list[LeaseView]] = {}
    for lease in active_now:
        if lease.kind == "hold":
            holds.setdefault(lease.role, []).append(lease)
        else:
            used[lease.role] = used.get(lease.role, 0) + 1
    hold_by_id = {h.lease_id: h for hs in holds.values() for h in hs}
    # A child only fills its hold's slot when both sit on the same role (a re-granted hold may
    # have moved while an old child finishes elsewhere; that child then just counts as used).
    busy = {l.hold_lease_id for l in active_now
            if l.hold_lease_id in hold_by_id and hold_by_id[l.hold_lease_id].role == l.role}

    ctx = _Ctx(cfg, roles, cards, now, used, holds, busy, set(), set())
    rank = cfg.priority_rank

    # --- 1. timeouts that need no placement -----------------------------------------
    queued: list[LeaseView] = []
    children: list[LeaseView] = []
    backlogged: list[LeaseView] = []
    for lease in leases:
        if lease.status == "retry_wait":
            if lease.deadline_at is not None and lease.deadline_at <= now:
                out.append(Unavailable(lease.lease_id, "deadline"))
            elif lease.hold_lease_id is not None and lease.hold_lease_id not in hold_by_id:
                out.append(Unavailable(lease.lease_id, "hold_not_granted"))   # never re-queue for a gone run
            elif lease.not_before is None or lease.not_before <= now:
                out.append(Requeue(lease.lease_id, "retry_due"))
                (children if lease.hold_lease_id else queued).append(lease)
            continue
        if lease.status == "queued" and lease.hold_lease_id is not None:
            if lease.deadline_at is not None and lease.deadline_at <= now:
                out.append(Unavailable(lease.lease_id, "deadline"))
            elif lease.hold_lease_id not in hold_by_id:
                out.append(Unavailable(lease.lease_id, "hold_not_granted"))
            else:
                children.append(lease)
            continue
        if lease.status == "queued":
            too_big = _exceeds_class(cfg, roles, lease, seen_ctx or {})
            if lease.deadline_at is not None and lease.deadline_at <= now:
                out.append(Unavailable(lease.lease_id, "deadline"))
            elif too_big is not None:
                # No card this class can use has a slot that big: say so now, naming the biggest,
                # instead of queueing until the deadline for a placement that can never happen.
                out.append(Unavailable(lease.lease_id, f"min_ctx_exceeds_class:{too_big}"))
            else:
                queued.append(lease)
        elif lease.status == "backlogged":
            # U4: a shed lease still ages out here -- deadlines apply under shed by design (plan
            # 2026-09-29 "Deadlines still apply"); only a very long cooling incident reaches it.
            if (now - lease.created_at).total_seconds() >= d.backlog_max_age_sec:
                out.append(DeadLetter(lease.lease_id, "backlog_max_age"))
            else:
                backlogged.append(lease)
        elif lease.status == "recalling" and lease.recall_by is not None and lease.recall_by <= now:
            out.append(Abort(lease.lease_id, PREEMPT) if lease.reason == PREEMPT else Abort(lease.lease_id))
        elif lease.status in ACTIVE and lease.expires_at is not None and lease.expires_at <= now:
            out.append(Expire(lease.lease_id))

    ctx.wanting = {c.hold_lease_id for c in children}

    # --- 2. what is draining this tick (no new grants there) -------------------------
    swap_roles = [r for r, spec in cfg.roles.items() if spec.swap is not None]
    frozen_seats = set(frozen or ()) | {r for r in swap_roles if cfg.roles[r].launch is None}
    max_held: set[str] = set()
    for seat in swap_roles:
        spec = cfg.roles[seat]
        if seat in frozen_seats:
            continue
        if ctx.loaded(seat):
            # A resident owner that could actually run there wants its card back: the seat drains.
            if not spec.operator_only and any(
                    ctx.cfg.owns(q.work_class, ev) and ctx.fits(q, ev)
                    for q in queued + backlogged if q.lease_id not in shed_of
                    for ev in cfg.evicted_by(seat)):
                ctx.draining.add(seat)
            # H4: a seat loaded for max_hold_sec gives its card back (loaded_at is only set when
            # the pool itself loaded or adopted the seat, never by observation alone).
            loaded_at = [cards[c].loaded_at for c in spec.cards if cards[c].loaded_at]
            if not spec.operator_only and spec.max_hold_sec and loaded_at \
                    and (now - min(loaded_at)).total_seconds() >= spec.max_hold_sec:
                ctx.draining.add(seat)
                max_held.add(seat)
        elif spec.operator_only and any(q.work_class in spec.owner and q.operator and q.lease_id not in shed_of
                                        for q in queued):
            # An operator is taking the cards: everything the seat evicts drains.
            ctx.draining.update(cfg.evicted_by(seat))

    # Backlogged work that something can serve again rejoins the queue in its original place.
    still_backlogged: list[LeaseView] = []
    for lease in backlogged:
        if any(ctx.placeable(lease, r) for r in cfg.classes[lease.work_class].roles):
            out.append(Requeue(lease.lease_id, "role_available"))
            queued.append(lease)
        else:
            still_backlogged.append(lease)
    backlogged = still_backlogged

    # U4: shed leases leave the placement pass entirely -- reported, never granted, never demand.
    # D3: a one-shot request is refused now (its caller falls back at once); the rest wait.
    for lease in _order(cfg, [l for l in queued if l.lease_id in shed_of]):
        if _one_shot(cfg, lease):
            out.append(Unavailable(lease.lease_id, f"shed:{shed_of[lease.lease_id]}"))
        else:
            out.append(Shed(lease.lease_id, f"shed:{shed_of[lease.lease_id]}"))
    queued = [l for l in queued if l.lease_id not in shed_of]
    backlogged = [l for l in backlogged if l.lease_id not in shed_of]

    # --- 3. grants: a run's own calls first, then owners on their own roles, then the rest
    order = _order(cfg, queued)
    granted: set[str] = set()
    granted_role: dict[str, str] = {}
    urgent_n = [sum(1 for l in active_now if l.hold_lease_id is None and l.priority == URGENT)]

    def capped(lease: LeaseView) -> bool:
        return lease.priority == URGENT and urgent_n[0] >= d.urgent_max_concurrent

    # Z1 queue order: a lease a role would take now but for serialize_with reserves the blocking
    # partner(s): no LATER lease in queue order is granted there, so a stream of overlapping calls
    # on one side cannot starve an older waiter on the other. role -> the role that reserved it.
    serial_reserved: dict[str, str] = {}

    def serial_blocked(lease: LeaseView, role: str) -> bool:
        """``lease`` could take ``role`` now except for serialize_with (Z1)."""
        return ctx.serialized_by(role) is not None and ctx.usable(role) and ctx.fits(lease, role) \
            and ctx.free_for(lease, role) > 0

    def reserve_partners(lease: LeaseView, roles: Iterable[str]) -> None:
        if capped(lease):
            return
        for role in roles:
            if serial_blocked(lease, role):
                for partner in cfg.serialized_with(role):
                    if ctx.used.get(partner) or ctx.holds.get(partner):
                        serial_reserved.setdefault(partner, role)

    def grant(lease: LeaseView, role: str) -> None:
        out.append(Grant(lease.lease_id, role))
        granted.add(lease.lease_id)
        granted_role[lease.lease_id] = role
        if lease.priority == URGENT and lease.hold_lease_id is None:
            urgent_n[0] += 1
        if lease.kind == "hold":
            holds.setdefault(role, []).append(lease)
            hold_by_id[lease.lease_id] = lease
        else:
            used[role] = used.get(role, 0) + 1
            if lease.hold_lease_id is not None:
                busy.add(lease.hold_lease_id)

    # H2: a child jumps its role's queue and needs only its hold's slot. Not `fits`: a recalled or
    # draining hold still finishes its current node inside the grace, and that needs its calls.
    # One slot per hold: while one of its calls is in flight the next waits for it, and never takes
    # a second slot ahead of the role's queue.
    for lease in sorted(children, key=lambda l: (l.created_at, l.lease_id)):
        hold = hold_by_id.get(lease.hold_lease_id)
        if hold is None or lease.hold_lease_id in busy:
            continue
        if ctx.usable(hold.role) and ctx.free_for(lease, hold.role) > 0:
            grant(lease, hold.role)

    # Roles whose owner already has work running there: a hold must not borrow them only to be
    # recalled on the next tick (the owner-demand recall below).
    owner_active = {l.role for l in active_now if l.hold_lease_id is None and cfg.owns(l.work_class, l.role)}

    for lease in order:
        if capped(lease):
            continue
        own = [r for r in cfg.classes[lease.work_class].roles if cfg.owns(lease.work_class, r)]
        for role in own:
            if role not in serial_reserved and ctx.placeable(lease, role) and ctx.free_for(lease, role) > 0:
                grant(lease, role)
                break
        else:
            reserve_partners(lease, own)

    def admitted_urgent() -> list[LeaseView]:
        """The waiting urgent leases the cap still has room for, in queue order. Only these are owed
        a slot: one past the cap is not a waiting owner and pauses nothing."""
        room = max(0, d.urgent_max_concurrent - urgent_n[0])
        return [q for q in order if q.priority == URGENT and q.lease_id not in granted
                and q.hold_lease_id is None][:room]

    admitted = {q.lease_id for q in admitted_urgent()}

    def owners_waiting(role: str) -> list[LeaseView]:
        # usable+fits, not placeable: an owner held back only by serialize_with (Z1) still waits for
        # its role, so no borrower may slip in ahead of it.
        return [q for q in order if q.lease_id not in granted
                and (q.priority != URGENT or q.lease_id in admitted)
                and cfg.owns(q.work_class, role) and ctx.usable(role) and ctx.fits(q, role)]

    for lease in order:
        if lease.lease_id in granted or capped(lease):
            continue
        for role in cfg.classes[lease.work_class].roles:
            if role in serial_reserved or not ctx.placeable(lease, role) or ctx.free_for(lease, role) <= 0:
                if role not in serial_reserved and (cfg.owns(lease.work_class, role) or not owners_waiting(role)):
                    reserve_partners(lease, [role])
                continue
            if not cfg.owns(lease.work_class, role) and owners_waiting(role):
                continue
            if lease.kind == "hold" and not cfg.owns(lease.work_class, role) and (
                    role in owner_active or any(granted_role.get(q.lease_id) == role
                                                and cfg.owns(q.work_class, role) for q in order)):
                continue
            grant(lease, role)
            break

    # serialize_with: say why a lease that a free role would otherwise take is still waiting --
    # blocked by an active partner, or by a role reserved for an older waiter on the other side.
    # Not for a lease the urgent cap holds back, nor on a borrowed role whose owners come first.
    for lease in order:
        if lease.lease_id in granted or capped(lease):
            continue
        for role in cfg.classes[lease.work_class].roles:
            if not cfg.owns(lease.work_class, role) and owners_waiting(role):
                continue
            blocker = ctx.serialized_by(role)
            if blocker is None and role in serial_reserved and ctx.usable(role) and ctx.fits(lease, role) \
                    and ctx.free_for(lease, role) > 0:
                blocker = serial_reserved[role]
            elif blocker is not None and not serial_blocked(lease, role):
                blocker = None
            if blocker is not None:
                out.append(Serialized(lease.lease_id, role, f"serialized:{blocker}"))
                break

    # --- 4. recalls ------------------------------------------------------------------
    recalled: set[str] = set()
    # A child is never recalled on its own: its hold is, and the child finishes inside that grace.
    active = [l for l in leases if l.status == "granted" and l.role and l.hold_lease_id is None]

    def recall(lease: LeaseView, reason: str) -> None:
        if lease.lease_id not in recalled:
            recalled.add(lease.lease_id)
            if reason == PREEMPT:
                grace = d.urgent_preempt_grace_sec
            else:
                grace = d.hold_clawback_grace_sec if lease.kind == "hold" else d.clawback_grace_sec
            out.append(Recall(lease.lease_id, now + timedelta(seconds=grace), reason))

    # U1 bookkeeping, before any recall: each waiting urgent lease (up to the cap) is owed one
    # pause. A pause already under way (recalling urgent_preempt) on a slot it could take pays
    # that debt first, so the short grace is not re-spent every tick (including the tick it runs
    # out and the hold is aborted). Greedy in queue order: <= urgent_max_concurrent leases.
    # Recounted after this tick's grants, which took cap room.
    waiting_urgent = admitted_urgent()
    admitted = {q.lease_id for q in waiting_urgent}
    pausing = [l for l in leases if l.status == "recalling" and l.reason == PREEMPT and l.role
               and l.hold_lease_id is None]
    # Past the abort, a paused hold's call still in flight keeps the slot until it ends (durable-runs
    # beats within the grace, then cancels it ~1 s after the abort). That call is still the pause: without this the urgent lease, not yet
    # granted, would pause another hold every tick until it did. Matched on the call's role.
    requeued = {l.lease_id: l for l in leases if l.status == "queued" and l.reason == PREEMPT
                and l.kind == "hold" and l.hold_lease_id is None}
    pausing += [replace(requeued[c.hold_lease_id], role=c.role) for c in active_now
                if c.hold_lease_id in requeued and c.hold_lease_id not in hold_by_id]

    def preemptible(l: LeaseView) -> bool:
        return l.kind == "hold" and not l.operator and l.priority in PREEMPTIBLE \
            and not (l.expires_at is not None and l.expires_at <= now)

    def could_take(u: LeaseView, v: LeaseView) -> bool:
        """Would ``u`` get ``v``'s slot once ``v`` is aborted and re-queued in place? Only on a role
        ``u``'s class may use, and not when ``u`` only borrows a role ``v`` owns: owners are granted
        first, so ``v`` would win it back and be paused again every grace. (A non-owner hold on a
        role with owner demand is recalled by the owner rules below, so U1 never sees it.)"""
        return v.role in cfg.classes[u.work_class].roles and ctx.placeable(u, v.role) \
            and (cfg.owns(u.work_class, v.role) or not cfg.owns(v.work_class, v.role))

    owed: list[LeaseView] = []
    for u in waiting_urgent:
        serving = next((p for p in pausing if could_take(u, p)), None)
        if serving is not None:
            pausing.remove(serving)
        else:
            owed.append(u)

    def reclaim(lease: LeaseView) -> None:
        """Owner demand takes a borrower back. When a waiting urgent owner is owed a pause and the
        borrower is pausable, this IS that pause: the short grace, re-queued in place (U2)."""
        if lease.lease_id in recalled:
            return
        u = next((u for u in owed if cfg.owns(u.work_class, lease.role) and could_take(u, lease)), None) \
            if preemptible(lease) else None
        if u is None:
            recall(lease, "owner_waiting")
        else:
            owed.remove(u)
            recall(lease, PREEMPT)

    for role in cfg.roles:
        starving = owners_waiting(role)
        if not starving:
            continue
        # Borrowers already giving the slot back count toward the owners waiting for it; without
        # this one waiting owner would recall one more borrower every tick of the grace period.
        already = sum(1 for l in leases if l.status == "recalling" and l.role == role
                      and l.hold_lease_id is None and not cfg.owns(l.work_class, role))
        needed = len(starving) - already
        if needed <= 0:
            continue
        borrowers = sorted(
            (l for l in active if l.role == role and not cfg.owns(l.work_class, role)),
            key=lambda l: (l.granted_at or l.created_at), reverse=True,
        )
        for lease in borrowers[:needed]:
            reclaim(lease)

    # A hold borrowing someone else's role gives it back once the owner has ANY demand there --
    # also when that demand is being served through the hold's gaps right now (H3), which would
    # otherwise hide the owner from `owners_waiting` for as long as the run keeps pausing.
    owner_demand: set[str] = set()
    for l in leases:
        if l.hold_lease_id is not None:
            continue
        role = granted_role.get(l.lease_id) or (l.role if l.status in ACTIVE else None)
        if role and cfg.owns(l.work_class, role):
            owner_demand.add(role)
    for role in cfg.roles:
        if owners_waiting(role):
            owner_demand.add(role)
    for lease in active:
        if lease.kind == "hold" and lease.role in owner_demand and not cfg.owns(lease.work_class, lease.role):
            reclaim(lease)

    for lease in active:
        if not cfg.owns(lease.work_class, lease.role):
            if any(not cards[c].lent for c in cfg.lendable_cards(lease.role)):
                recall(lease, "card_unlent")
        if lease.role in max_held:
            recall(lease, "max_hold")
        elif lease.role in ctx.draining:
            recall(lease, "draining")
        spec = cfg.roles[lease.role]
        if lease.kind == "hold" and not lease.operator and spec.swap is None and spec.max_hold_sec \
                and lease.granted_at and (now - lease.granted_at).total_seconds() >= spec.max_hold_sec:
            recall(lease, "max_hold")

    # U1: each urgent lease still owed a pause pauses one preemptible hold it could replace:
    # background before system, most recently granted first.
    victims = sorted((l for l in active if preemptible(l)),
                     key=lambda l: (-rank(l.priority), -(l.granted_at or l.created_at).timestamp(), l.lease_id))
    unserved_urgent: set[str] = set()
    for u in owed:
        victim = next((v for v in victims if v.lease_id not in recalled and could_take(u, v)), None)
        if victim is None:
            unserved_urgent.add(u.lease_id)
        else:
            recall(victim, PREEMPT)

    # --- 5. nothing can serve it: wait / backlog / fail ------------------------------
    def reclaimable(role: str) -> bool:
        """Evicted by a non-operator swap seat: the owner's own demand drains that seat."""
        return any(cfg.roles[s].swap is not None and not cfg.roles[s].operator_only
                   and ctx.loaded(s) and role in cfg.evicted_by(s) for s in cfg.roles)

    def serviceable(lease: LeaseView) -> bool:
        for role in cfg.classes[lease.work_class].roles:
            if ctx.fits(lease, role) and (ctx.usable(role) or _loadable(ctx, role)):
                return True
            if role in ctx.draining and role not in max_held and ctx.roles.get(role) \
                    and ctx.roles[role].healthy:
                return True  # it comes back after the drain
            if cfg.owns(lease.work_class, role) and reclaimable(role):
                return True  # waiting for its card back is not "nothing can serve it"
        return False

    for lease in order:
        if lease.lease_id in granted or serviceable(lease):
            continue
        policy = cfg.classes[lease.work_class].on_unavailable
        if policy == "backlog" and not lease.retryable:
            policy = "wait"  # a backlog nobody comes back for would only hand slots to ghosts
        if policy == "backlog":
            out.append(Backlog(lease.lease_id, "no_serviceable_role"))
        elif policy == "fail":
            out.append(Unavailable(lease.lease_id, "no_serviceable_role"))

    # --- 6. swap seats ----------------------------------------------------------------
    waiting = [q for q in order if q.lease_id not in granted] + backlogged
    for seat in swap_roles:
        spec = cfg.roles[seat]
        seat_cards = [cards[c] for c in spec.cards]
        if any(c.swap_state != "idle" for c in seat_cards):
            continue
        evicted = cfg.evicted_by(seat)
        if ctx.loaded(seat):
            busy_seat = ctx.occupancy(seat) > 0
            if seat in ctx.draining and not busy_seat:
                out.append(SwapUnload(seat, "max_hold" if seat in max_held else "owner_reclaim"))
            elif spec.operator_only:
                holders = [l for l in leases if l.work_class in spec.owner and l.status in ACTIVE + ("queued",)]
                over = spec.max_hold_sec and any(
                    l.granted_at and (now - l.granted_at).total_seconds() >= spec.max_hold_sec for l in holders)
                if not holders:
                    out.append(SwapUnload(seat, "operator_released"))
                elif over and seat not in frozen_seats:   # frozen: its unload cannot happen, keep the holders
                    for l in holders:
                        if l.status == "granted":
                            recall(l, "max_hold")
            elif not busy_seat and not any(seat in cfg.classes[q.work_class].roles for q in waiting):
                last = max((c.last_active_at for c in seat_cards if c.last_active_at), default=None)
                if last is None or (now - last).total_seconds() >= d.swap_idle_unload_sec:
                    out.append(SwapUnload(seat, "idle"))
            continue
        if spec.operator_only:
            wanting = [q for q in order if q.work_class in spec.owner and q.operator]
            if wanting and all(ctx.occupancy(r) == 0 for r in evicted):
                out.append(SwapLoad(seat, "operator"))
            continue
        # U3: urgent work no pause can serve justifies a load at once; guards below still apply.
        waited = [q for q in waiting if seat in cfg.classes[q.work_class].roles
                  and (q.lease_id in unserved_urgent
                       or (now - (q.queued_since or q.created_at)).total_seconds() >= cfg.swap_after_wait_sec(seat))]
        fit = {q.lease_id: ctx.fits_for_load(q, seat, seen_ctx or {}) for q in waited}
        wanting = [q for q in waited if fit[q.lease_id]]
        unknown = [q for q in waited if fit[q.lease_id] is None]
        residents_idle = all(ctx.occupancy(r) == 0 for r in evicted)
        residents_wanted = any(cfg.owns(q.work_class, r) for q in waiting for r in evicted)
        if residents_idle and not residents_wanted and unknown and not wanting:
            # Demand the seat could serve, but its context size was never seen: say so, not silence.
            out.append(SwapBlocked(seat, "ctx_unknown",
                                   f"min_ctx_tokens={max(q.min_ctx_tokens for q in unknown)}"))
            continue
        if not (wanting and residents_idle and not residents_wanted):
            continue
        blocked = _load_blocked(spec.swap.guards if spec.swap else [], seat_cards, now, guards)
        out.append(SwapBlocked(seat, *blocked) if blocked else SwapLoad(seat, "demand"))
    return out


def _load_blocked(seat_guards: list[str], seat_cards: list[CardLive], now: datetime,
                  guards: dict[str, str | None] | None) -> tuple[str, str | None] | None:
    """S1: why a justified load may not start yet, or None."""
    if any(c.residency_until and c.residency_until > now for c in seat_cards):
        return "min_residency", None
    if any(c.cooldown_until and c.cooldown_until > now for c in seat_cards):
        return "cooldown", None
    if guards is None:
        return None
    for name in seat_guards:
        state = guards.get(name, "unavailable")
        if state:
            return f"guard:{name}", state
    return None


def _exceeds_class(cfg: PoolConfig, roles: dict[str, RoleLive], lease: LeaseView,
                   seen_ctx: dict[str, int]) -> int | None:
    """The largest known per-slot context of the class's LLM roles, when it is smaller than the
    lease needs; else None. Known = live now, or last seen (a role restarting keeps its size). A
    role never seen (a swap seat not yet loaded) might be bigger, so no known context means
    "cannot tell yet", never "too big"."""
    if not lease.min_ctx_tokens:
        return None
    known = []
    for r in cfg.classes[lease.work_class].roles:
        if cfg.roles[r].kind != "llm":
            continue
        size = (roles[r].ctx_per_slot if r in roles else None) or seen_ctx.get(r)
        if size:
            known.append(size)
    if not known or max(known) >= lease.min_ctx_tokens:
        return None
    return max(known)


def _loadable(ctx: _Ctx, role: str) -> bool:
    """A swap seat that is not loaded can still serve work once loaded."""
    spec = ctx.cfg.roles[role]
    if spec.swap is None or ctx.loaded(role) or ctx.evicted(role):
        return False
    return all(ctx.cards[c].swap_state == "idle" for c in spec.cards)
