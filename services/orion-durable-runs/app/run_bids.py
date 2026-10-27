"""Durable-runs' own bid site (body economy Patch C3a, SHADOW: logged, never obeyed).

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

Gated by ORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED (and needs the market: ORION_PHYSIOLOGY_BID_ENABLED).
Best-effort: never raises into admission.
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.run_acts import run_act_family
from orion.schemas.physiology_want import WantV1, WantWindowV1

logger = logging.getLogger(__name__)

PRODUCER = "orion-durable-runs.admission"
BID_TIMEOUT_SEC = 30.0          # a cold history read is a 30-day scan; the run never waits on it
SEEN_MAX = 5000


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


def want_for_run(request: dict[str, Any], tier: int) -> Optional[WantV1]:
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
        want_id=run_id, family=family.actor, producer=PRODUCER, verb=family.verb,
        object_ref=f"durable_run:{run_id}", tier=tier, opened_at=_aware(request.get("requested_at")),
        window=WantWindowV1(latest=_aware(admission.get("deadline_at"))),
        correlation_id=str(corr) if corr else None)


class RunBidder:
    """``market_getter`` returns the process's ``Market`` (or None while the market is off)."""

    def __init__(self, *, enabled: bool, market_getter: Callable[[], Any],
                 timeout_sec: float = BID_TIMEOUT_SEC) -> None:
        self.enabled = bool(enabled)
        self._market = market_getter
        self._timeout = timeout_sec
        self._tasks: set[asyncio.Task] = set()
        self._seen: OrderedDict[str, None] = OrderedDict()
        # Runs whose bid is still in flight, and what happened to them meanwhile: an open (its time)
        # or the terminal. Applied when the bid lands, so a fast run never leaks a reservation.
        self._inflight: dict[str, dict[str, Any]] = {}
        self.counts = {"bid": 0, "skipped": 0, "failed": 0, "opened": 0, "released": 0}

    def _market_or_none(self) -> Any:
        if not self.enabled:
            return None
        try:
            return self._market()
        except Exception:  # noqa: BLE001
            return None

    def on_submit(self, request: dict[str, Any]) -> Optional[asyncio.Task]:
        """Schedule the shadow bid for an accepted run. Returns the task (tests await it), or None."""
        market = self._market_or_none()
        if market is None or getattr(market, "families", None) is None:
            return None
        try:
            run_id = str(request.get("run_id") or "")
            if not run_id or run_id in self._seen:
                return None          # a duplicate submit of the same run bids once
            family = run_act_family(request)
            if family is None:
                return None
            cfg = market.families.get(family.actor)
            want = want_for_run(request, cfg.tier if cfg is not None else 0)
            if want is None:
                self.counts["skipped"] += 1
                logger.info("physiology_shadow_bid_skipped run=%s family=%s reason=%s", run_id, family.actor,
                            "not_in_families_yaml" if cfg is None else f"tier_{cfg.tier}")
                return None
            self._seen[run_id] = None
            while len(self._seen) > SEEN_MAX:
                self._seen.popitem(last=False)
            self._inflight[run_id] = {}
            task = asyncio.get_running_loop().create_task(self._bid(market, want))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return task
        except Exception:  # noqa: BLE001 -- never block admission on the economy
            self.counts["failed"] += 1
            logger.warning("physiology_shadow_bid_schedule_failed run=%s", request.get("run_id"), exc_info=True)
            return None

    async def _bid(self, market: Any, want: WantV1) -> Any:
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
        seen = self._inflight.pop(want.want_id, {})
        if seen.get("ended"):
            self.counts["released"] += self._safe(market.release, want.want_id)
        elif seen.get("opened_at") is not None:
            self.counts["opened"] += self._safe(market.mark_opened, want.want_id, seen["opened_at"])
        logger.info("physiology_shadow_bid run=%s corr=%s family=%s decision_id=%s outcome=%s reason=%s "
                    "binding=%s shadow=true obeyed=false", want.want_id, want.correlation_id, want.family,
                    decision.decision_id, decision.outcome, decision.reason, decision.binding_constraint)
        return decision

    @staticmethod
    def _safe(fn: Callable[..., Any], *args: Any) -> int:
        try:
            return int(fn(*args) or 0)
        except Exception:  # noqa: BLE001
            logger.warning("physiology_shadow_bid_book_call_failed", exc_info=True)
            return 0

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


__all__ = ["PRODUCER", "RunBidder", "want_for_run"]
