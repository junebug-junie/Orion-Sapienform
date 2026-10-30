"""The dream cycle's BINDING bid (body economy Patch D for dream.cycle). The sleep obeys its bid.

Before this, a check that would sleep placed a SHADOW bid (app/physiology_shadow.py, Patch C3b) and
never obeyed it. With ``ORION_PHYSIOLOGY_BID_DREAM_CYCLE=on``, a check that has passed every one of the
dream's own gates (pressure or the overdue backstop, the refractory interval, arousal, and something to
replay) asks the market just before the sleep opens its act, and obeys the answer with the same rule
curiosity investigate obeys (``orion.physiology.binding.bind_verdict``, #3010):

* funded -> sleep now;
* the metacog lane is full but the sleep is worth its price (``capacity_reserved``, net > 0) -> sleep
  now: its LLM calls queue at the GPU pool, which is the queue;
* deferred on price (Prime's slow charge counts here), no hunger yet, or net <= 0 -> HOLD: no sleep this
  check, and the want bids again at the first check at or after the market's ``retry_after_sec``
  (clamped to 2 min .. 1 h; checks run every ``DREAM_CYCLE_CHECK_INTERVAL_SEC``);
* no answer (market down, stale body state, history unread, timeout) -> sleep now, as before. This
  patch removes no other gate, so "no answer" falls back to the dream's own gates, never to "held".

Bounds: a held sleep is never dropped and never held past ``hold_end``, the EARLIER of

* the want's FIRST bid + the family's deferrable window (``DEFERRABLE_WINDOW_S``, 6 h; the same number as
  config/physiology/families.yaml ``dream.cycle``, asserted by a test). From the first bid, not from the
  first check that wanted to sleep: the dream's own arousal hold often runs for hours first and must not use
  up the economy's window; and
* the overdue line: the last good sleep's start + ``DREAM_LOOKBACK_HOURS`` (48 h). Past it, replay
  material starts falling out of the window, so a sleep that is overdue (the backstop) never bids at
  all: it sleeps (``floor:overdue``). The economy can delay consolidation; it cannot starve it.

The want. One want per stretch between sleeps: the first check that clears the dream's gates opens it
(cycle id minted then), and every later check re-uses the same cycle id until a sleep attempt ends, so
the want_ledger, the market's Prime record and the eventual act all carry one id. Held in process: a
restart opens a new want (and a new 6 h window), so the 6 h bound is per process lifetime; across repeated
restarts the bound is the overdue line (48 h), which survives a restart.

Every verdict is one log line (``physiology_bind ... obeyed=true``) with the market's own reason, net,
charge and ``why``; the decision itself is in ``want_ledger`` with producer ``BINDING_PRODUCER``. A sleep
that ran after a hold says so in its cycle note.

Forced runs (``POST /dreams/cycle/run?force=true``) never bid: the operator's call, not the dream's want.

Flag: ``ORION_PHYSIOLOGY_BID_DREAM_CYCLE``: ``on`` (shipped) binds; ``shadow`` (or anything else) ->
the C3b shadow bid, exactly as before. Rollback = set ``shadow`` and restart.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

from orion.physiology.binding import BindVerdict, bind_verdict

logger = logging.getLogger("orion-dream.physiology_bind")

FAMILY = "dream.cycle"
VERB = "make"
TIER = 1
# Binding bids carry their own producer, so want_ledger (whose ``shadow`` is always true) tells an
# obeyed decision from a shadow one (scripts/analysis/body_economy_shadow_vs_real.py BINDING_PRODUCERS).
BINDING_PRODUCER = "orion-dream.dream_cycle.binding"
# config/physiology/families.yaml dream.cycle deferrable_window_s. The image does not ship families.yaml;
# tests/test_physiology_bind.py fails if the two drift.
DEFERRABLE_WINDOW_S = 21600.0
# The sleep waits on this bid (a cold market history read is a 30-day scan). A timeout sleeps as before.
BIND_TIMEOUT_SEC = 5.0
MODES = ("on", "shadow")
# Strong references to in-flight release tasks (asyncio keeps only weak ones).
_TASKS: set[asyncio.Task] = set()


def bind_mode(raw: Any = None) -> str:
    """``on`` or ``shadow``. Unknown values are shadow (logged), never a crash."""
    if raw is None:
        try:
            from app.settings import settings

            raw = getattr(settings, "ORION_PHYSIOLOGY_BID_DREAM_CYCLE", "on")
        except Exception:  # noqa: BLE001
            raw = "on"
    mode = str(raw or "").strip().lower()
    if mode not in MODES:
        logger.warning("physiology_bind_unknown_mode value=%r -> shadow", raw)
        return "shadow"
    return mode


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def overdue_line(last_window_start: Optional[datetime], lookback_hours: float) -> Optional[datetime]:
    """When unreplayed material starts falling out of view (app.cycle.overdue). None: never slept."""
    last = _utc(last_window_start)
    return None if last is None else last + timedelta(hours=float(lookback_hours))


def hold_end(want_opened_at: datetime, last_window_start: Optional[datetime], *, lookback_hours: float,
             window_s: float = DEFERRABLE_WINDOW_S) -> datetime:
    """The latest a held sleep may wait: the earlier of the want's window and the overdue line. Pure."""
    ends = [_utc(want_opened_at) + timedelta(seconds=float(window_s))]
    line = overdue_line(last_window_start, lookback_hours)
    if line is not None:
        ends.append(line)
    return min(ends)


def next_check_at(now: datetime, retry_at: datetime, check_interval_sec: float) -> datetime:
    """The first sleep-loop check at or after ``retry_at`` (checks run every ``check_interval_sec``)."""
    step = max(1.0, float(check_interval_sec))
    n = max(1, math.ceil((retry_at - now).total_seconds() / step))
    return now + timedelta(seconds=n * step)


@dataclass
class HeldWant:
    """The open want of the stretch since the last sleep attempt."""
    cycle_id: str
    opened_at: datetime
    holds: int = 0
    retry_at: Optional[datetime] = None
    last_reason: str = ""
    first_why: str = ""
    last_why: str = ""
    funded: bool = False
    # The economy's window starts at the want's FIRST bid, not when it opened: checks the dream's own gates
    # held (arousal, mostly; live 10-29/30 that was hours before most sleeps) must not use it up.
    first_bid_at: Optional[datetime] = None
    # Set when a bind admits it: a second check racing on the same want (the loop and a manual run) is
    # refused instead of opening a second act with the same cycle id.
    admitted: bool = False


async def _abid(want: Any, timeout_sec: float) -> Any:
    """The client call, behind one name so tests can stub it. None on any failure."""
    from orion.physiology.client import abid

    return await abid(want, timeout_sec=timeout_sec)


async def _arelease(want_id: str) -> Any:
    from orion.physiology.client import arelease

    return await arelease(want_id)


class DreamBinder:
    """One process's dream want and its binding bids. Every method is best-effort and never raises."""

    def __init__(self, *, timeout_sec: float = BIND_TIMEOUT_SEC) -> None:
        self.want: Optional[HeldWant] = None
        self._timeout = timeout_sec
        self._lock: Optional[asyncio.Lock] = None
        self.counts = {"bid": 0, "admitted": 0, "held": 0, "no_answer": 0, "floor": 0, "window_end": 0}

    def current(self, now: datetime, last_attempt_end: Optional[datetime]) -> HeldWant:
        """The open want, or a new one. A sleep attempt that ended after the want opened closed it."""
        last = _utc(last_attempt_end)
        if self.want is not None and (self.want.admitted or (last is not None and last >= self.want.opened_at)):
            self.want = None
        if self.want is None:
            self.want = HeldWant(cycle_id=f"dc-{uuid4().hex[:12]}", opened_at=now)
        return self.want

    def close(self) -> None:
        """The sleep attempt ran (or ended empty): the next want is a new one."""
        self.want = None

    async def bind(self, want: HeldWant, now: datetime, *, last_window_start: Optional[datetime],
                   lookback_hours: float, check_interval_sec: float, overdue: bool) -> BindVerdict:
        """The verdict for this check. Never raises; any failure is an ``admit`` (today's behaviour).

        Serialized per process: the sleep loop and a manual run may hold the same want. The first to be
        admitted takes it; the other is refused (``duplicate``) and opens nothing."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if want.admitted:
                v = BindVerdict(False, "duplicate", detail={"reason": "duplicate",
                                                            "why": "another check is already sleeping this want"})
                self._log(want, "-", v)
                return v
            try:
                v = await self._bind(want, now, last_window_start=last_window_start, lookback_hours=lookback_hours,
                                     check_interval_sec=check_interval_sec, overdue=overdue)
            except Exception:  # noqa: BLE001 -- never hold a sleep because this code failed
                logger.warning("physiology_bind_failed family=%s cycle_id=%s", FAMILY, want.cycle_id, exc_info=True)
                v = BindVerdict(True, "no_answer:bind_failed")
            if v.admit:
                want.admitted = True
            return v

    async def _bind(self, want: HeldWant, now: datetime, *, last_window_start: Optional[datetime],
                    lookback_hours: float, check_interval_sec: float, overdue: bool) -> BindVerdict:
        from app import act_emit

        want_id = act_emit.cycle_correlation_id(want.cycle_id)
        if overdue:
            # The floor: an overdue sleep never asks. Asking could only delay consolidation past the window
            # it already missed (and a charge would be credited with holding a sleep that had to happen).
            self.counts["floor"] += 1
            v = BindVerdict(True, "floor:overdue")
            self._log(want, want_id, v)
            return v
        if want.first_bid_at is None:
            want.first_bid_at = now
        end = hold_end(want.first_bid_at, last_window_start, lookback_hours=lookback_hours)
        if now >= end:
            self.counts["window_end"] += 1
            v = BindVerdict(True, "window_end", detail={"why": want.last_why} if want.last_why else {})
            self._log(want, want_id, v)
            return v
        if want.retry_at is not None and now < want.retry_at:
            # The market named a later relief time: hold without a bid (the same prices would answer).
            self.counts["held"] += 1
            v = BindVerdict(False, f"waiting:{want.last_reason}", retry_at=want.retry_at,
                            detail={"why": want.last_why})
            self._log(want, want_id, v)
            return v
        from orion.schemas.physiology_want import WantV1, WantWindowV1

        bid = WantV1(want_id=want_id, family=FAMILY, producer=BINDING_PRODUCER, verb=VERB, tier=TIER,
                     object_ref=f"dream_cycle:{want.cycle_id}", correlation_id=want_id, opened_at=want.opened_at,
                     window=WantWindowV1(latest=end))
        decision = None
        try:
            decision = await asyncio.wait_for(_abid(bid, self._timeout), self._timeout + 1.0)
        except Exception:  # noqa: BLE001 -- incl. timeout: sleep as before
            logger.warning("physiology_bind_bid_failed family=%s want_id=%s", FAMILY, want_id, exc_info=True)
        if decision is not None:
            self.counts["bid"] += 1
        v = bind_verdict(decision, now, end)
        want.funded = bool(decision is not None and getattr(decision, "outcome", None) == "funded")
        if decision is not None and not want.funded:
            # A held or pool-queued want reserves nothing. Best effort, off the sleep's path.
            task = asyncio.get_running_loop().create_task(_arelease(want_id))
            _TASKS.add(task)
            task.add_done_callback(_TASKS.discard)
        why = str(v.detail.get("why") or "")
        if v.admit:
            self.counts["no_answer" if v.why.startswith("no_answer") else "admitted"] += 1
        else:
            self.counts["held"] += 1
            want.holds += 1
            want.retry_at = next_check_at(now, v.retry_at, check_interval_sec) if v.retry_at else None
            want.last_reason = str(v.detail.get("reason") or v.why)
            want.first_why = want.first_why or why
        want.last_why = why or want.last_why
        self._log(want, want_id, v)
        return v

    async def release_after_sleep(self, want: Optional[HeldWant]) -> None:
        """The sleep's act closed: free a funded reservation now instead of at expiry. Best effort."""
        if want is None or not want.funded:
            return
        from app import act_emit

        try:
            await asyncio.wait_for(_arelease(act_emit.cycle_correlation_id(want.cycle_id)), 2.0)
        except Exception:  # noqa: BLE001
            logger.info("physiology_bind_release_failed cycle_id=%s", want.cycle_id)

    @staticmethod
    def _log(want: HeldWant, want_id: str, v: BindVerdict) -> None:
        d = v.detail
        logger.info(
            "physiology_bind family=%s want_id=%s cycle_id=%s verdict=%s admit=%s holds=%d retry_at=%s "
            "decision_id=%s outcome=%s reason=%s binding=%s net=%s charge=%s why=%r obeyed=true",
            FAMILY, want_id, want.cycle_id, v.why, v.admit, want.holds,
            want.retry_at.isoformat() if (want.retry_at and not v.admit) else None, d.get("decision_id"),
            d.get("outcome"), d.get("reason"), d.get("binding_constraint"), d.get("net"), d.get("charge"),
            str(d.get("why") or "")[:300],
        )


def held_note(want: Optional[HeldWant], now: datetime) -> str:
    """The cycle note's line for a sleep that the economy held first ('' when it never held)."""
    if want is None or want.holds <= 0:
        return ""
    mins = (now - (want.first_bid_at or want.opened_at)).total_seconds() / 60.0
    return (f" | body economy: held {want.holds} time(s) over {mins:.0f} min before this sleep; "
            f"first why: {want.first_why[:200]}")


# One per process: the sleep loop and the manual endpoint share the open want.
BINDER = DreamBinder()

__all__ = ["BINDER", "BINDING_PRODUCER", "DEFERRABLE_WINDOW_S", "DreamBinder", "FAMILY", "HeldWant", "bind_mode",
           "held_note", "hold_end", "next_check_at", "overdue_line"]
