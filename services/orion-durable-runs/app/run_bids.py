"""Durable-runs' own bid site (body economy Patch C3a SHADOW; Patch D BINDING for curiosity investigate).

Patch D (2026-10-30): runs of ``BINDING_FAMILIES`` (curiosity investigate only) do not bid at
submit. They bid at their first ``resource_request`` (``RunBidder.bind``, before the run asks the
GPU pool for a hold) and OBEY the answer -- see ``bind_verdict`` for the rule. A held run records a
``run.economy_held`` event with the market's own reason (net, charge, binding constraint, why) and
bids again later. Flag: ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE (``on`` binds; ``shadow`` -> the
C3a shadow bid at submit, exactly as before). Every other family below is still shadow.

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Execution seam" and
Phase 1 Patch C ("every producer bids in shadow: it logs a would-fund decision next to its real
gate decision").

Most of Orion's chosen work never passes a cortex verb dispatch: curiosity investigate,
self-inquiry, self-sense eval and the durable journal are durable runs admitted here. So this
process bids for them itself, in-process (no HTTP call to its own ``/physiology/bid``):

* **submit** (the run is accepted into admission): a run in a want family -- exactly the set
  ``app.run_acts.run_act_family`` opens acts for -- becomes a ``WantV1`` and is decided by the
  market (``Market.bid``, which publishes the decision to ``want_ledger``). The bid runs as a
  background task: ``submit`` returns exactly as before, whatever the market says or however long
  it takes.
* **first grant** (the act opens): ``Market.mark_opened`` starts the want's reservation, one
  act-length from the real start. The bid reserves PENDING (``reserve_at_open``): runs wait a median
  1.3-1.6 h in the pool's queue, longer than one act-length, so a reservation from the decision
  would hold the lane while the run waits and lapse before it runs.
* **terminal** (the act closes, or the run ends without ever running): ``Market.release`` ends the
  reservation, so a shadow-funded want no real act uses stops biasing later bids toward deferral.

Joins. ``want_id`` = the run id (also the act's ``object_ref`` ``durable_run:<run_id>``);
``correlation_id`` = the run's correlation id, which is the act's ``correlation_id`` in
``grammar_acts`` (live 10-27: run ids and correlation ids differ, so the run id alone does not join).

Attention patch 3b (docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md,
section 1): the GPU pool serves its background queue by value. So every decision here, shadow or
binding, is remembered per run (``value_for``) and its net rides on the run's hold request. Reading
(``reading.turn``, which opens no act here: Hub opens it) gets a VALUE-ONLY bid at stage 1's submit:
the decision is logged and its net is carried, but it reserves nothing and nothing obeys it (binding
reading is patch 3c). Stage 2 never bids: it carries its stage 1's net (one job, one bid). A run whose
bid has not landed, or a process that restarted, carries no net: the pool keeps it in arrival order.

Gated by ORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED (and needs the market: ORION_PHYSIOLOGY_BID_ENABLED).
Best-effort: never raises into admission.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from app.run_acts import run_act_family
# The pure rule is shared with orion-dream (the dream cycle's binding bid): one rule, two bid sites.
from orion.physiology.binding import (  # noqa: F401 -- re-exported: tests, replays and evals import them here
    NO_ANSWER_REASONS,
    REBID_DEFAULT_SEC,
    REBID_MAX_SEC,
    REBID_MIN_SEC,
    BindVerdict,
    bind_verdict,
    decision_detail,
)
from orion.schemas.physiology_want import WantV1, WantWindowV1
from orion.schemas.reading_turn import READING_WORKFLOW

logger = logging.getLogger(__name__)

PRODUCER = "orion-durable-runs.admission"
# Binding bids carry their own producer, so want_ledger (whose ``shadow`` is always true) still tells
# an obeyed decision from a shadow one.
BINDING_PRODUCER = "orion-durable-runs.admission.binding"
BID_TIMEOUT_SEC = 30.0          # shadow: a cold history read is a 30-day scan; the run never waits on it
# Binding: the run DOES wait on this bid (it holds a driver slot), so it is short; a timeout asks the
# pool as before.
BIND_TIMEOUT_SEC = 5.0
SEEN_MAX = 5000

# --- Patch D: binding bids (one family per PR; curiosity investigate first) -----------------------
# A run in a BINDING family asks the market before it asks the GPU pool for a hold. Funded: it asks
# the pool, as before. Deferred: it is HELD (status waiting_resource, no pool request) and bids again
# later. It is never dropped: at the end of its family's deferrable window it asks the pool anyway,
# exactly as it would have without the economy. No answer from the market (down, stale state, history
# unread, timeout): it asks the pool, as before -- this patch removes no other gate (Hub's own vetoes,
# the pool's queue and the heat reflex still apply), so "no answer" falls back to the old gates, never
# to "funded by the economy" and never to "held forever".
BINDING_FAMILIES = frozenset({"durable.curiosity_investigate"})
HOLD_DEADLINE_MARGIN_SEC = 1800.0   # a run with its own deadline is released this long before it
HOLD_EVENT = "run.economy_held"
# Patch 3b: reading's value-only bid (stage 1). The act is Hub's ``hub.world_pulse_read look``.
READING_FAMILY = "hub.world_pulse_read"
READING_VERB = "look"
RELEASE_EVENT = "run.economy_admitted"


def _aware(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(v, datetime):
        return None
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


def want_for_run(request: dict[str, Any], tier: int, *, producer: str = PRODUCER) -> Optional[WantV1]:
    """The run's want, or None when the run is not one of Orion's own wants (or not a bidding tier)."""
    family = run_act_family(request)
    if family is None or tier not in (1, 2):
        return None
    run_id = str(request.get("run_id") or "")
    if not run_id:
        return None
    admission = request.get("admission") if isinstance(request.get("admission"), dict) else {}
    corr = request.get("correlation_id")
    return WantV1(
        want_id=run_id, family=family.actor, producer=producer, verb=family.verb,
        object_ref=f"durable_run:{run_id}", tier=tier, opened_at=_aware(request.get("requested_at")),
        window=WantWindowV1(latest=_aware(admission.get("deadline_at"))),
        correlation_id=str(corr) if corr else None)


@dataclass(frozen=True)
class RunValue:
    """What a run's hold request carries to the pool (patch 3b): the market decision's want, net and
    family. The pool orders its background class by ``net``."""
    want_id: str
    net: float
    family: str


def reading_stage(request: dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
    """(seed_id, stage) of a reading run, else (None, None)."""
    if request.get("workflow") != READING_WORKFLOW:
        return None, None
    brief = request.get("brief") if isinstance(request.get("brief"), dict) else {}
    seed, stage = brief.get("seed_id"), brief.get("stage")
    return (str(seed), int(stage)) if seed and stage in (1, 2) else (None, None)


def reading_want(request: dict[str, Any], tier: int) -> Optional[WantV1]:
    """Stage 1's value-only want (patch 3b), or None (not reading, stage 2, or no run id)."""
    seed, stage = reading_stage(request)
    run_id = str(request.get("run_id") or "")
    if seed is None or stage != 1 or not run_id or tier not in (1, 2):
        return None
    corr = request.get("correlation_id")
    return WantV1(want_id=run_id, family=READING_FAMILY, producer=PRODUCER, verb=READING_VERB,
                  object_ref=f"durable_run:{run_id}", tier=tier, opened_at=_aware(request.get("requested_at")),
                  correlation_id=str(corr) if corr else None)


def hold_end(want: WantV1, deferrable_window_s: float) -> Optional[datetime]:
    """When a held want stops waiting and asks the pool anyway: ``opened_at`` + the family's
    deferrable window, but never later than ``HOLD_DEADLINE_MARGIN_SEC`` before the run's own
    admission deadline (a run released AT its deadline would fail at once). None: no opened_at and
    no deadline -- the caller treats that as "do not hold" (an unbounded hold is never allowed)."""
    ends = []
    if want.opened_at is not None and deferrable_window_s > 0:
        ends.append(want.opened_at + timedelta(seconds=deferrable_window_s))
    if want.window.latest is not None:
        ends.append(want.window.latest - timedelta(seconds=HOLD_DEADLINE_MARGIN_SEC))
    return min(ends) if ends else None


def binding_family(request: dict[str, Any], enabled: bool) -> Optional[str]:
    """The run's family when its bid binds (flag on, the family is in BINDING_FAMILIES, and the run is
    not an operator's forced run -- Hub "Run now" sets ``brief.operator_forced``: the operator's call,
    not curiosity's want, so it is never held), else None."""
    if not enabled:
        return None
    brief = request.get("brief") if isinstance(request.get("brief"), dict) else {}
    if brief.get("operator_forced"):
        return None
    fam = run_act_family(request)
    return fam.actor if fam is not None and fam.actor in BINDING_FAMILIES else None


class RunBidder:
    """``market_getter`` returns the process's ``Market`` (or None while the market is off)."""

    def __init__(self, *, enabled: bool, market_getter: Callable[[], Any],
                 timeout_sec: float = BID_TIMEOUT_SEC, binding_enabled: bool = False,
                 bind_timeout_sec: float = BIND_TIMEOUT_SEC) -> None:
        self.enabled = bool(enabled)
        # Patch D: BINDING_FAMILIES' runs bid at their first resource_request and obey (``bind``);
        # off -> those runs bid in shadow at submit like every other family.
        self.binding_enabled = bool(binding_enabled)
        self._bind_timeout = bind_timeout_sec
        self._market = market_getter
        self._timeout = timeout_sec
        self._tasks: set[asyncio.Task] = set()
        self._seen: OrderedDict[str, None] = OrderedDict()
        # Runs whose bid is still in flight, and what happened to them meanwhile: an open (its time)
        # or the terminal. Applied when the bid lands, so a fast run never leaks a reservation.
        self._inflight: dict[str, dict[str, Any]] = {}
        self.counts = {"bid": 0, "skipped": 0, "failed": 0, "opened": 0, "released": 0,
                       "bind_admitted": 0, "bind_held": 0, "bind_no_answer": 0, "value_bid": 0}
        # Patch 3b: run_id -> the net its hold carries; reading seed -> stage 1's (stage 2 reuses it).
        self._values: OrderedDict[str, RunValue] = OrderedDict()
        self._seed_values: OrderedDict[str, RunValue] = OrderedDict()
        self._task_of: dict[str, asyncio.Task] = {}

    def _market_or_none(self) -> Any:
        if not (self.enabled or self.binding_enabled):
            return None
        try:
            return self._market()
        except Exception:  # noqa: BLE001
            return None

    def on_submit(self, request: dict[str, Any]) -> Optional[asyncio.Task]:
        """Schedule the shadow bid for an accepted run. Returns the task (tests await it), or None."""
        if not self.enabled:
            return None
        market = self._market_or_none()
        if market is None or getattr(market, "families", None) is None:
            return None
        try:
            if binding_family(request, self.binding_enabled) is not None:
                return None          # its binding bid (``bind``, at resource_request) is its one bid
            run_id = str(request.get("run_id") or "")
            if not run_id or run_id in self._seen:
                return None          # a duplicate submit of the same run bids once
            family = run_act_family(request)
            value_only = False
            if family is None:
                if reading_stage(request)[1] != 1:
                    return None
                cfg = market.families.get(READING_FAMILY)
                want = reading_want(request, cfg.tier if cfg is not None else 0)
                value_only = True
                if want is None:
                    return None
            else:
                cfg = market.families.get(family.actor)
                want = want_for_run(request, cfg.tier if cfg is not None else 0)
            if want is None:
                self.counts["skipped"] += 1
                logger.info("physiology_shadow_bid_skipped run=%s family=%s reason=%s", run_id,
                            family.actor if family is not None else READING_FAMILY,
                            "not_in_families_yaml" if cfg is None else f"tier_{cfg.tier}")
                return None
            self._seen[run_id] = None
            while len(self._seen) > SEEN_MAX:
                self._seen.popitem(last=False)
            self._inflight[run_id] = {}
            task = asyncio.get_running_loop().create_task(self._bid(market, want, request, value_only=value_only))
            self._tasks.add(task)
            self._task_of[run_id] = task
            task.add_done_callback(self._tasks.discard)
            task.add_done_callback(lambda _t, r=run_id: self._task_of.pop(r, None))
            return task
        except Exception:  # noqa: BLE001 -- never block admission on the economy
            self.counts["failed"] += 1
            logger.warning("physiology_shadow_bid_schedule_failed run=%s", request.get("run_id"), exc_info=True)
            return None

    async def _bid(self, market: Any, want: WantV1, request: Optional[dict[str, Any]] = None, *,
                   value_only: bool = False) -> Any:
        try:
            decision = await asyncio.wait_for(market.bid(want, reserve_at_open=True), self._timeout)
        except Exception:  # noqa: BLE001 -- incl. timeout: the run proceeds exactly as before
            self.counts["failed"] += 1
            logger.warning("physiology_shadow_bid_failed run=%s family=%s", want.want_id, want.family, exc_info=True)
            # A timed-out bid may still land in the book: release it, it will never be opened.
            self._inflight.pop(want.want_id, None)
            self._safe(market.release, want.want_id)
            return None
        self.counts["bid"] += 1
        self._remember(request or {}, want, decision)
        seen = self._inflight.pop(want.want_id, {})
        if value_only:
            # Reading's bid only sets its queue value (patch 3b): it reserves nothing. Binding reading,
            # with its reservations, is patch 3c.
            self.counts["value_bid"] += 1
            self._safe(market.release, want.want_id)
        elif seen.get("ended"):
            self.counts["released"] += self._safe(market.release, want.want_id)
        elif seen.get("opened_at") is not None:
            self.counts["opened"] += self._safe(market.mark_opened, want.want_id, seen["opened_at"])
        logger.info("physiology_shadow_bid run=%s corr=%s family=%s decision_id=%s outcome=%s reason=%s "
                    "binding=%s net=%s value_only=%s shadow=true obeyed=false", want.want_id, want.correlation_id,
                    want.family, decision.decision_id, decision.outcome, decision.reason, decision.binding_constraint,
                    getattr(decision, "net", None), value_only)
        return decision

    def _remember(self, request: dict[str, Any], want: WantV1, decision: Any) -> None:
        """Keep the decision's net for the run's hold request (and, for reading stage 1, its seed)."""
        net = getattr(decision, "net", None)
        if net is None:
            return
        v = RunValue(want.want_id, float(net), want.family)
        self._values[v.want_id] = v
        seed, _stage = reading_stage(request)
        if seed is not None:
            self._seed_values[seed] = v
        for m in (self._values, self._seed_values):
            while len(m) > SEEN_MAX:
                m.popitem(last=False)

    async def value_for(self, request: dict[str, Any], wait_sec: float = BIND_TIMEOUT_SEC) -> Optional[RunValue]:
        """The net this run's hold request carries (patch 3b), or None (no bid: arrival order). Waits up
        to ``wait_sec`` for a shadow bid still in flight, so the first hold request carries it. A
        reading stage 2 carries its stage 1's net. Never raises."""
        try:
            run_id = str(request.get("run_id") or "")
            task = self._task_of.get(run_id)
            if task is not None and not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), wait_sec)
                except Exception:  # noqa: BLE001 -- incl. timeout: ask the pool without a net
                    pass
            if run_id in self._values:
                return self._values[run_id]
            seed, stage = reading_stage(request)
            return self._seed_values.get(seed) if seed is not None and stage == 2 else None
        except Exception:  # noqa: BLE001
            logger.warning("physiology_run_value_failed run=%s", request.get("run_id"), exc_info=True)
            return None

    @staticmethod
    def _safe(fn: Callable[..., Any], *args: Any) -> int:
        try:
            return int(fn(*args) or 0)
        except Exception:  # noqa: BLE001
            logger.warning("physiology_shadow_bid_book_call_failed", exc_info=True)
            return 0

    async def bind(self, request: dict[str, Any], now: datetime) -> Optional[BindVerdict]:
        """Patch D: the binding bid of a run that has not asked the pool yet. None when the run's family
        does not bind (flag off, or not in BINDING_FAMILIES): admission goes on exactly as before.

        Never raises. No market / timeout / failure -> an ``admit`` verdict with ``no_answer:*``."""
        try:
            family = binding_family(request, self.binding_enabled)
        except Exception:  # noqa: BLE001
            logger.warning("physiology_bind_family_failed run=%s", request.get("run_id"), exc_info=True)
            return None
        if family is None:
            return None
        market = self._market_or_none()
        cfg = market.families.get(family) if market is not None and getattr(market, "families", None) else None
        if cfg is None:
            self.counts["bind_no_answer"] += 1
            v = BindVerdict(True, "no_answer:market_off" if market is None else "no_answer:no_families_config")
            self._log(request, v)
            return v
        want = want_for_run(request, cfg.tier, producer=BINDING_PRODUCER)
        if want is None:
            return None
        decision = None
        t0 = time.monotonic()
        try:
            decision = await asyncio.wait_for(market.bid(want, now, reserve_at_open=True), self._bind_timeout)
        except Exception:  # noqa: BLE001 -- incl. timeout: the run asks the pool as before
            logger.warning("physiology_bind_bid_failed run=%s family=%s", want.want_id, want.family, exc_info=True)
        elapsed_ms = round((time.monotonic() - t0) * 1000.0, 1)
        verdict = bind_verdict(decision, now, hold_end(want, cfg.deferrable_window_s))
        verdict.detail["bid_ms"] = elapsed_ms
        if decision is None or decision.outcome != "funded":
            # Nothing of this want may stay in the book: a held or pool-queued want reserves nothing
            # (its use shows in the lane's measured level). wait_for cancels a timed-out bid; the
            # release is a no-op then.
            self._safe(market.release, want.want_id)
        if decision is not None:
            self.counts["bid"] += 1
            self._remember(request, want, decision)
        key = "bind_held" if not verdict.admit else (
            "bind_no_answer" if verdict.why.startswith("no_answer") else "bind_admitted")
        self.counts[key] += 1
        self._log(request, verdict)
        return verdict

    @staticmethod
    def _log(request: dict[str, Any], v: BindVerdict) -> None:
        d = v.detail
        logger.info("physiology_bind run=%s corr=%s verdict=%s admit=%s retry_at=%s decision_id=%s outcome=%s "
                    "reason=%s binding=%s net=%s charge=%s bid_ms=%s obeyed=true", request.get("run_id"),
                    request.get("correlation_id"), v.why, v.admit, v.retry_at.isoformat() if v.retry_at else None,
                    d.get("decision_id"), d.get("outcome"), d.get("reason"), d.get("binding_constraint"),
                    d.get("net"), d.get("charge"), d.get("bid_ms"))

    def on_grant(self, run_id: str, at: Optional[datetime] = None) -> int:
        """The run got a pool grant (its act opens at the first one): start its reservation."""
        market = self._market_or_none()
        if market is None or not run_id:
            return 0
        pending = self._inflight.get(str(run_id))
        if pending is not None and pending.get("opened_at") is None:
            pending["opened_at"] = at or datetime.now(timezone.utc)   # applied when the bid lands
        try:
            n = market.mark_opened(str(run_id), at)
        except Exception:  # noqa: BLE001
            logger.warning("physiology_shadow_bid_open_failed run=%s", run_id, exc_info=True)
            return 0
        self.counts["opened"] += n
        return n

    def on_terminal(self, run_id: str) -> int:
        """The run ended (its act closed, or it never ran): release its reservation now."""
        market = self._market_or_none()
        if market is None or not run_id:
            return 0
        if str(run_id) in self._inflight:
            self._inflight[str(run_id)]["ended"] = True              # applied when the bid lands
        try:
            n = market.release(str(run_id))
        except Exception:  # noqa: BLE001
            logger.warning("physiology_shadow_bid_release_failed run=%s", run_id, exc_info=True)
            return 0
        self.counts["released"] += n
        return n


__all__ = ["BINDING_FAMILIES", "BINDING_PRODUCER", "BindVerdict", "HOLD_EVENT", "NO_ANSWER_REASONS", "PRODUCER",
           "READING_FAMILY", "REBID_DEFAULT_SEC", "REBID_MAX_SEC", "REBID_MIN_SEC", "RELEASE_EVENT", "RunBidder",
           "RunValue", "bind_verdict", "binding_family", "decision_detail", "hold_end", "reading_stage",
           "reading_want", "want_for_run"]
