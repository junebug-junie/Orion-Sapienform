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
    swap_state: str = "idle"      # idle | loading | unloading
    cooldown_until: datetime | None = None   # set when a swap seat unloads; blocks reload
    last_active_at: datetime | None = None   # last time a lease held a swap seat on this card


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


Decision = Union[Grant, Recall, Abort, Expire, Unavailable, Backlog, Requeue, DeadLetter, SwapLoad, SwapUnload]


@dataclass
class _Ctx:
    cfg: PoolConfig
    roles: dict[str, RoleLive]
    cards: dict[str, CardLive]
    now: datetime
    occupancy: dict[str, int]
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

    def usable(self, role: str) -> bool:
        live = self.roles.get(role)
        return bool(live and live.healthy and live.slots > 0 and self.loaded(role))

    def free(self, role: str) -> int:
        live = self.roles.get(role)
        return (live.slots if live else 0) - self.occupancy.get(role, 0)

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
) -> list[Decision]:
    d = cfg.defaults
    out: list[Decision] = []

    occupancy: dict[str, int] = {}
    for lease in leases:
        if lease.status in ACTIVE and lease.role and not (
                lease.expires_at is not None and lease.expires_at <= now):
            occupancy[lease.role] = occupancy.get(lease.role, 0) + 1

    ctx = _Ctx(cfg, roles, cards, now, occupancy, set(), set())

    # --- 1. timeouts that need no placement -----------------------------------------
    queued: list[LeaseView] = []
    backlogged: list[LeaseView] = []
    for lease in leases:
        if lease.status == "retry_wait":
            if lease.deadline_at is not None and lease.deadline_at <= now:
                out.append(Unavailable(lease.lease_id, "deadline"))
            elif lease.not_before is None or lease.not_before <= now:
                out.append(Requeue(lease.lease_id, "retry_due"))
                queued.append(lease)
            continue
        if lease.status == "queued":
            if lease.deadline_at is not None and lease.deadline_at <= now:
                out.append(Unavailable(lease.lease_id, "deadline"))
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
    for seat in swap_roles:
        spec = cfg.roles[seat]
        if ctx.loaded(seat):
            # A resident owner that could actually run there wants its card back: the seat drains.
            if not spec.operator_only and any(
                    ctx.cfg.owns(q.work_class, ev) and ctx.fits(q, ev)
                    for q in queued + backlogged for ev in cfg.evicted_by(seat)):
                ctx.draining.add(seat)
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

    # --- 3. grants: owners on their own roles first, then everyone else -------------
    order = _order(cfg, queued)
    granted: set[str] = set()

    def grant(lease: LeaseView, role: str) -> None:
        out.append(Grant(lease.lease_id, role))
        granted.add(lease.lease_id)
        occupancy[role] = occupancy.get(role, 0) + 1

    for lease in order:
        for role in cfg.classes[lease.work_class].roles:
            if cfg.owns(lease.work_class, role) and ctx.placeable(lease, role) and ctx.free(role) > 0:
                grant(lease, role)
                break

    def owners_waiting(role: str) -> list[LeaseView]:
        return [q for q in order if q.lease_id not in granted
                and cfg.owns(q.work_class, role) and ctx.placeable(q, role)]

    for lease in order:
        if lease.lease_id in granted:
            continue
        for role in cfg.classes[lease.work_class].roles:
            if not ctx.placeable(lease, role) or ctx.free(role) <= 0:
                continue
            if not cfg.owns(lease.work_class, role) and owners_waiting(role):
                continue
            grant(lease, role)
            break

    # --- 4. recalls ------------------------------------------------------------------
    recall_by = now + timedelta(seconds=d.clawback_grace_sec)
    recalled: set[str] = set()
    active = [l for l in leases if l.status == "granted" and l.role]

    def recall(lease: LeaseView, reason: str) -> None:
        if lease.lease_id not in recalled:
            recalled.add(lease.lease_id)
            out.append(Recall(lease.lease_id, recall_by, reason))

    for role in cfg.roles:
        starving = owners_waiting(role)
        if not starving:
            continue
        # Borrowers already giving the slot back count toward the owners waiting for it; without
        # this one waiting owner would recall one more borrower every tick of the grace period.
        already = sum(1 for l in leases if l.status == "recalling" and l.role == role
                      and not cfg.owns(l.work_class, role))
        needed = len(starving) - already
        if needed <= 0:
            continue
        borrowers = sorted(
            (l for l in active if l.role == role and not cfg.owns(l.work_class, role)),
            key=lambda l: (l.granted_at or l.created_at), reverse=True,
        )
        for lease in borrowers[:needed]:
            recall(lease, "owner_waiting")

    for lease in active:
        if not cfg.owns(lease.work_class, lease.role):
            if any(not cards[c].lent for c in cfg.lendable_cards(lease.role)):
                recall(lease, "card_unlent")
        if lease.role in ctx.draining:
            recall(lease, "draining")

    # --- 5. nothing can serve it: wait / backlog / fail ------------------------------
    def reclaimable(role: str) -> bool:
        """Evicted by a non-operator swap seat: the owner's own demand drains that seat."""
        return any(cfg.roles[s].swap is not None and not cfg.roles[s].operator_only
                   and ctx.loaded(s) and role in cfg.evicted_by(s) for s in cfg.roles)

    def serviceable(lease: LeaseView) -> bool:
        for role in cfg.classes[lease.work_class].roles:
            if ctx.fits(lease, role) and (ctx.usable(role) or _loadable(ctx, role)):
                return True
            if role in ctx.draining and ctx.roles.get(role) and ctx.roles[role].healthy:
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
            busy = occupancy.get(seat, 0) > 0
            if seat in ctx.draining and not busy:
                out.append(SwapUnload(seat, "owner_reclaim"))
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
            elif not busy and not any(seat in cfg.classes[q.work_class].roles for q in waiting):
                last = max((c.last_active_at for c in seat_cards if c.last_active_at), default=None)
                if last is None or (now - last).total_seconds() >= d.swap_idle_unload_sec:
                    out.append(SwapUnload(seat, "idle"))
            continue
        if any(c.cooldown_until and c.cooldown_until > now for c in seat_cards) and not spec.operator_only:
            continue
        if spec.operator_only:
            wanting = [q for q in order if q.work_class in spec.owner and q.operator]
            if wanting and all(occupancy.get(r, 0) == 0 for r in evicted):
                out.append(SwapLoad(seat, "operator"))
            continue
        wanting = [
            q for q in waiting
            if seat in cfg.classes[q.work_class].roles and ctx.fits(q, seat)
            and (now - (q.queued_since or q.created_at)).total_seconds() >= d.swap_after_wait_sec
        ]
        residents_idle = all(occupancy.get(r, 0) == 0 for r in evicted)
        residents_wanted = any(cfg.owns(q.work_class, r) for q in waiting for r in evicted)
        if wanting and residents_idle and not residents_wanted:
            out.append(SwapLoad(seat, "demand"))
    return out


def _loadable(ctx: _Ctx, role: str) -> bool:
    """A swap seat that is not loaded can still serve work once loaded."""
    spec = ctx.cfg.roles[role]
    if spec.swap is None or ctx.loaded(role) or ctx.evicted(role):
        return False
    return all(ctx.cards[c].swap_state == "idle" for c in spec.cards)
