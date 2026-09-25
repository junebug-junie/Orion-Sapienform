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
  H1 a hold (kind="hold") is placed like any lease, at most one per role, and reserves one slot
  H2 a child (hold_lease_id set) runs in its hold's slot: only on the hold's role, ahead of the
     role's queue, never behind its own run, never taking a second slot for the pair
  H3 interleave: while a hold has no child in flight, a lease of STRICTLY higher priority may use
     that slot (gaps are shared, Juniper 2026-09-25); equal/lower priority and other holds may not
  H4 hold recall uses defaults.hold_clawback_grace_sec; a swap seat with max_hold_sec drains once
     it has been loaded that long, then unloads (today's DURABLE_RUNS_ELASTIC_MAX_BORROW_SEC)
  S1 a seat load is blocked -- reported as SwapBlocked, never silent -- by min residency after an
     unload, cooldown after a failed load, or a failing guard (thermal, visual_baseline)
  S2 a card in "fault" grants nothing on any of its roles; a card mid-swap grants nothing on the
     seat or the roles it evicts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Union

from orion.gpu_pool.config import PoolConfig

ACTIVE = ("granted", "recalling")


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


Decision = Union[Grant, Recall, Abort, Expire, Unavailable, Backlog, Requeue, DeadLetter, SwapLoad, SwapUnload,
                 SwapBlocked]


@dataclass
class _Ctx:
    cfg: PoolConfig
    roles: dict[str, RoleLive]
    cards: dict[str, CardLive]
    now: datetime
    used: dict[str, int]                  # active requests + children, per role
    holds: dict[str, list[LeaseView]]     # active holds, per role (at most one each)
    busy_holds: set[str]                  # holds with a child in flight (the child sits in `used`)
    draining: set[str]
    operator_granted: set[str]

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
        return [h for h in self.holds.get(role, []) if h.lease_id not in self.busy_holds]

    def occupancy(self, role: str) -> int:
        """Slots taken, counting a hold with no child in flight as holding its one slot."""
        return self.used.get(role, 0) + len(self.idle_holds(role))

    def free_for(self, lease: LeaseView, role: str) -> int:
        """H1-H3: free slots on ``role`` as ``lease`` sees them."""
        if lease.kind == "hold" and self.holds.get(role):
            return 0  # at most one hold per role
        live = self.roles.get(role)
        n = (live.slots if live else 0) - self.used.get(role, 0)
        rank = self.cfg.priority_rank
        for hold in self.idle_holds(role):
            if lease.hold_lease_id == hold.lease_id:
                continue  # a child runs in its own hold's slot
            if lease.kind != "hold" and lease.hold_lease_id is None \
                    and rank(lease.priority) < rank(hold.priority):
                continue  # strictly higher priority may use the gap between the run's calls
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

    def placeable(self, lease: LeaseView, role: str) -> bool:
        return self.usable(role) and self.fits(lease, role)


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
) -> list[Decision]:
    """``seen_ctx``: each role's last-seen per-slot context, kept by the caller across restarts of
    the role. Used ONLY to decide "too big for this class" -- a briefly-down big role must not make
    its class look small. Placement, swaps and serviceability use live ``roles`` alone.

    ``guards``: swap-guard name -> None when clear, else why it fails (a guard the caller could not
    read must be passed as failing, e.g. "unavailable"). A name missing from a dict fails closed.
    ``None`` means the caller evaluates no guards at all (pure tests, the replay eval)."""
    d = cfg.defaults
    out: list[Decision] = []

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
            if (now - lease.created_at).total_seconds() >= d.backlog_max_age_sec:
                out.append(DeadLetter(lease.lease_id, "backlog_max_age"))
            else:
                backlogged.append(lease)
        elif lease.status == "recalling" and lease.recall_by is not None and lease.recall_by <= now:
            out.append(Abort(lease.lease_id))
        elif lease.status in ACTIVE and lease.expires_at is not None and lease.expires_at <= now:
            out.append(Expire(lease.lease_id))

    # --- 2. what is draining this tick (no new grants there) -------------------------
    swap_roles = [r for r, spec in cfg.roles.items() if spec.swap is not None]
    max_held: set[str] = set()
    for seat in swap_roles:
        spec = cfg.roles[seat]
        if ctx.loaded(seat):
            # A resident owner that could actually run there wants its card back: the seat drains.
            if not spec.operator_only and any(
                    ctx.cfg.owns(q.work_class, ev) and ctx.fits(q, ev)
                    for q in queued + backlogged for ev in cfg.evicted_by(seat)):
                ctx.draining.add(seat)
            # H4: a seat loaded for max_hold_sec gives its card back (loaded_at is only set when
            # the pool itself loaded or adopted the seat, never by observation alone).
            loaded_at = [cards[c].loaded_at for c in spec.cards if cards[c].loaded_at]
            if not spec.operator_only and spec.max_hold_sec and loaded_at \
                    and (now - min(loaded_at)).total_seconds() >= spec.max_hold_sec:
                ctx.draining.add(seat)
                max_held.add(seat)
        elif spec.operator_only and any(q.work_class in spec.owner and q.operator for q in queued):
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

    # --- 3. grants: a run's own calls first, then owners on their own roles, then the rest
    order = _order(cfg, queued)
    granted: set[str] = set()
    granted_role: dict[str, str] = {}

    def grant(lease: LeaseView, role: str) -> None:
        out.append(Grant(lease.lease_id, role))
        granted.add(lease.lease_id)
        granted_role[lease.lease_id] = role
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
        for role in cfg.classes[lease.work_class].roles:
            if cfg.owns(lease.work_class, role) and ctx.placeable(lease, role) and ctx.free_for(lease, role) > 0:
                grant(lease, role)
                break

    def owners_waiting(role: str) -> list[LeaseView]:
        return [q for q in order if q.lease_id not in granted
                and cfg.owns(q.work_class, role) and ctx.placeable(q, role)]

    for lease in order:
        if lease.lease_id in granted:
            continue
        for role in cfg.classes[lease.work_class].roles:
            if not ctx.placeable(lease, role) or ctx.free_for(lease, role) <= 0:
                continue
            if not cfg.owns(lease.work_class, role) and owners_waiting(role):
                continue
            if lease.kind == "hold" and not cfg.owns(lease.work_class, role) and (
                    role in owner_active or any(granted_role.get(q.lease_id) == role
                                                and cfg.owns(q.work_class, role) for q in order)):
                continue
            grant(lease, role)
            break

    # --- 4. recalls ------------------------------------------------------------------
    recalled: set[str] = set()
    # A child is never recalled on its own: its hold is, and the child finishes inside that grace.
    active = [l for l in leases if l.status == "granted" and l.role and l.hold_lease_id is None]

    def recall(lease: LeaseView, reason: str) -> None:
        if lease.lease_id not in recalled:
            recalled.add(lease.lease_id)
            grace = d.hold_clawback_grace_sec if lease.kind == "hold" else d.clawback_grace_sec
            out.append(Recall(lease.lease_id, now + timedelta(seconds=grace), reason))

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
            recall(lease, "owner_waiting")

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
            recall(lease, "owner_waiting")

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
                elif over:
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
        wanting = [
            q for q in waiting
            if seat in cfg.classes[q.work_class].roles and ctx.fits(q, seat)
            and (now - (q.queued_since or q.created_at)).total_seconds() >= cfg.swap_after_wait_sec(seat)
        ]
        residents_idle = all(ctx.occupancy(r) == 0 for r in evicted)
        residents_wanted = any(cfg.owns(q.work_class, r) for q in waiting for r in evicted)
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
