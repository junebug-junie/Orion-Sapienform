"""Hub outreach's BINDING bid (body economy Patch D for ``hub.outreach``). Outreach obeys its bid.

Before this, the background tick placed a SHADOW bid at its compose point (scripts/physiology_shadow.py,
Patch C3b; once per window under strain, #2988) and never obeyed it; the reading, self-finding and office
doors did not bid at all. With ``ORION_PHYSIOLOGY_BID_HUB_OUTREACH=on``, every door that composes a
message in Hub asks the market just before it composes, and obeys the answer with the same rule
curiosity investigate (#3010) and the dream cycle obey (``orion.physiology.binding.bind_verdict``):

* funded -> compose now;
* deferred because a lane is full (``capacity_reserved``) but worth its price (net > 0) -> compose now:
  the GPU pool's own queue is the queue;
* deferred on price, no hunger yet (``sated``), or net <= 0 -> HOLD: no compose this check. The want
  bids again at the first check at or after the market's ``retry_after_sec`` (clamped to 2 min .. 1 h);
  checks before that hold without a bid (one bid per window, never one per 10 s tick);
* no answer (market down, stale body state, history unread, a 5 s timeout, any error) -> compose now,
  as before. This patch removes no other gate, so "no answer" falls back to the old gates, never to
  "held".

Bounds: a held want is never dropped. Its window is ``DEFERRABLE_WINDOW_S`` (1 h, config/physiology/
families.yaml ``hub.outreach``, asserted by a test) from its FIRST bid; at the end it composes anyway
(``window_end``). The doors keep their own pending queues (a held reading finding, self finding or office
visit stays pending and is retried on the door's next tick), so a hold only delays.

Doors (``services/orion-hub/scripts/endogenous_outreach.py``):

* background tick (``endogenous_tick``): binds; cap-counted (charges ``attention:juniper``, 1 message).
* reading findings, self findings, office visits: bind; cap-EXEMPT (Juniper's decisions, 2026-10-13,
  11-03, 11-05: no daily cap), so their want says ``juniper_msgs=0``: they draw nothing on the cap line
  the body measures (orion-durable-runs ``physiology_store.CAP_EXEMPT_SOURCES``).
* curiosity Door-A: does NOT bid. Its message is composed inside the curiosity run, under that run's
  own GPU hold, and that run already obeyed its own binding bid (#3010); its send is cap-exempt; and a
  blocked Door-A is final (no retry path), so a hold would drop the want.
* strain report: does NOT bid. It speaks only when ``arousal_strained`` is the one gate holding, to tell
  Juniper about the strain itself. Its strain-held SHADOW bid (#2988, once per window) is unchanged.
* forced (``POST /api/debug/endogenous-outreach/trigger``): never held, never bids (operator's call).

While the ``arousal_strained`` outreach veto stays (it is deleted in the follow-up PR, after 7 days of
this reader ``on``), a strained tick never reaches its compose point, so the binding bid never decides
during strain: the strain-held shadow bid is the only record of what the market would say then.

The want. One want per door key (the tick: one; a door: one per bundle / visit). Held in process: a
restart opens a new want and a new 1 h window. An admitted want stays "admitted" until the door's
attempt ends (``close``): a second check racing on the same want in that time is refused
(``duplicate``) instead of composing twice.

One lock serializes every door's bid (a bid waits at most ~6 s behind another door's): simple, and the
duplicate guard needs it. Every verdict that bids, admits or ends a window is one log line (``physiology_bind family=hub.outreach ... obeyed=true``) with the
market's reason, binding constraint, net, charge and ``why``; the decision row
(``endogenous_outreach_decisions.result_json.economy``) carries the same; the decision itself is in
``want_ledger`` with producer ``BINDING_PRODUCER`` (orion.physiology.binding.BINDING_PRODUCERS).

Flag: ``ORION_PHYSIOLOGY_BID_HUB_OUTREACH``: ``on`` (shipped) binds; ``shadow`` -> the C3b shadow bid at
the tick's compose point, exactly as before, and no door bids. Anything else is shadow (logged once).
Rollback = set ``shadow`` and restart Hub.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from orion.physiology.binding import BindVerdict, bind_verdict

logger = logging.getLogger("orion-hub.outreach_bind")

FAMILY = "hub.outreach"
VERB = "say"
TIER = 2
# Binding bids carry their own producer, so want_ledger (whose ``shadow`` is always true) tells an
# obeyed decision from a shadow one (orion/physiology/binding.py BINDING_PRODUCERS).
BINDING_PRODUCER = "orion-hub.outreach.binding"
# config/physiology/families.yaml hub.outreach deferrable_window_s. The image does not read families.yaml
# here; tests/test_outreach_bind.py fails if the two drift.
DEFERRABLE_WINDOW_S = 3600.0
# The compose waits on this bid (a cold market history read is a 30-day scan). A timeout composes as before.
BIND_TIMEOUT_SEC = 5.0
MODES = ("on", "shadow")
HELD_REASON = "economy_held"
DUPLICATE_REASON = "economy_duplicate"
TICK_KEY = "endogenous_tick"
# A held want nobody re-checked for this long is forgotten (a bundle a door dropped, a visit that aged
# out). Its next check, if any, opens a new want.
FORGET_AFTER_SEC = 24 * 3600.0
# A held want not re-checked within this long after its relief time has LAPSED (the door stopped asking:
# no content, quiet hours, a human gate). Its next check opens a new want with a fresh bid and a fresh
# window, never a free ``window_end`` pass hours later (review finding 1). Above the slowest door tick (300 s).
LAPSE_AFTER_SEC = 900.0
# An admitted want whose attempt never closed (a crash between bind and close) stops refusing after this.
ADMITTED_STALE_SEC = 2 * DEFERRABLE_WINDOW_S
# Strong references to in-flight release tasks (asyncio keeps only weak ones).
_TASKS: set[asyncio.Task] = set()


def bind_mode(raw: Any = None) -> str:
    """``on`` or ``shadow``. Unknown values are shadow (logged), never a crash."""
    if raw is None:
        try:
            from scripts.settings import settings

            raw = getattr(settings, "ORION_PHYSIOLOGY_BID_HUB_OUTREACH", "on")
        except Exception:  # noqa: BLE001
            raw = "on"
    mode = str(raw or "").strip().lower()
    if mode not in MODES:
        logger.warning("physiology_bind_unknown_mode family=%s value=%r -> shadow", FAMILY, raw)
        return "shadow"
    return mode


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class HeldWant:
    """One door key's open want."""
    key: str
    door: str
    want_id: str
    opened_at: datetime
    first_bid_at: Optional[datetime] = None
    last_check_at: Optional[datetime] = None
    holds: int = 0
    bids: int = 0
    retry_at: Optional[datetime] = None
    last_reason: str = ""
    first_why: str = ""
    last_why: str = ""
    funded: bool = False
    admitted: bool = False
    admitted_at: Optional[datetime] = None


async def _abid(want: Any, timeout_sec: float) -> Any:
    """The client call, behind one name so tests can stub it. None on any failure."""
    from orion.physiology.client import abid

    return await abid(want, timeout_sec=timeout_sec)


async def _arelease(want_id: str) -> Any:
    from orion.physiology.client import arelease

    return await arelease(want_id)


def _release_soon(want_id: str) -> None:
    try:
        task = asyncio.get_running_loop().create_task(_arelease(want_id))
        _TASKS.add(task)
        task.add_done_callback(_TASKS.discard)
    except Exception:  # noqa: BLE001 -- best effort
        logger.info("physiology_bind_release_not_started want_id=%s", want_id)


class OutreachBinder:
    """Hub outreach's open wants and their binding bids. Every method is best-effort and never raises."""

    def __init__(self, *, timeout_sec: float = BIND_TIMEOUT_SEC, window_s: float = DEFERRABLE_WINDOW_S) -> None:
        self.wants: Dict[str, HeldWant] = {}
        self._timeout = float(timeout_sec)
        self._window = float(window_s)
        self._lock: Optional[asyncio.Lock] = None
        self.counts = {"bid": 0, "admitted": 0, "held": 0, "no_answer": 0, "window_end": 0, "forced": 0,
                       "duplicate": 0}

    def end(self, want: HeldWant) -> Optional[datetime]:
        return None if want.first_bid_at is None else want.first_bid_at + timedelta(seconds=self._window)

    async def bind(self, key: str, *, door: str, now: Optional[datetime] = None, cap_counted: bool = True,
                   forced: bool = False) -> BindVerdict:
        """The verdict for this check. Never raises; any failure is an ``admit`` (today's behaviour).

        Serialized per process: the first check admitted on a want takes it until ``close``; another
        check on the same want meanwhile is refused (``duplicate``) and composes nothing."""
        now = _utc(now or datetime.now(timezone.utc))
        if forced:
            self.counts["forced"] += 1
            v = BindVerdict(True, "forced", detail={"why": "operator-forced send: never held, never bids"})
            self._log(key, door, None, v)
            return v
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            self._forget(now)
            want = self.wants.get(key)
            if (want is not None and not want.admitted and want.retry_at is not None
                    and (now - want.retry_at).total_seconds() > LAPSE_AFTER_SEC):
                self.wants.pop(key, None)   # lapsed: the door stopped asking; a new want bids afresh
                want = None
            if want is not None and want.admitted:
                if want.admitted_at is not None and (now - want.admitted_at).total_seconds() > ADMITTED_STALE_SEC:
                    self.wants.pop(key, None)   # its attempt never closed: a lost close, not a live race
                    want = None
                else:
                    self.counts["duplicate"] += 1
                    v = BindVerdict(False, "duplicate", detail={
                        "reason": "duplicate", "why": "another check is already composing this want"})
                    self._log(key, door, want, v)
                    return v
            if want is None:
                want = HeldWant(key=key, door=door, want_id=str(uuid4()), opened_at=now)
                self.wants[key] = want
            try:
                v = await self._bind(want, now, cap_counted=cap_counted)
            except Exception:  # noqa: BLE001 -- never hold a message because this code failed
                logger.warning("physiology_bind_failed family=%s door=%s key=%s", FAMILY, door, key, exc_info=True)
                v = BindVerdict(True, "no_answer:bind_failed")
            want.last_check_at = now
            if v.admit:
                want.admitted, want.admitted_at = True, now
            return v

    async def _bind(self, want: HeldWant, now: datetime, *, cap_counted: bool) -> BindVerdict:
        if want.first_bid_at is None:
            want.first_bid_at = now
        end = self.end(want)
        if now >= end:
            self.counts["window_end"] += 1
            v = BindVerdict(True, "window_end", detail={"why": want.last_why} if want.last_why else {})
            self._log(want.key, want.door, want, v)
            return v
        if want.retry_at is not None and now < want.retry_at:
            # The market named a later relief time: hold without a bid (the same prices would answer).
            # Not logged: the tick checks every 10 s; the bid that set the hold was logged.
            self.counts["held"] += 1
            return BindVerdict(False, f"waiting:{want.last_reason}", retry_at=want.retry_at,
                               detail={"why": want.last_why, "reason": want.last_reason})
        from orion.schemas.grammar import ActCostV2
        from orion.schemas.physiology_want import WantV1, WantWindowV1

        # A cap-exempt door's message draws nothing on attention:juniper as the body measures it.
        cost = None if cap_counted else ActCostV2(juniper_msgs=0)
        bid = WantV1(want_id=want.want_id, family=FAMILY, producer=BINDING_PRODUCER, verb=VERB, tier=TIER,
                     object_ref="juniper", correlation_id=want.want_id, opened_at=want.opened_at,
                     window=WantWindowV1(latest=end), cost=cost)
        decision = None
        try:
            decision = await asyncio.wait_for(_abid(bid, self._timeout), self._timeout + 1.0)
        except Exception:  # noqa: BLE001 -- incl. timeout: compose as before
            logger.warning("physiology_bind_bid_failed family=%s want_id=%s", FAMILY, want.want_id, exc_info=True)
        if decision is None:
            # A timed-out bid may still land in the market's book (the HTTP call runs on in its thread):
            # release it, as run_bids.py does (review finding 4).
            _release_soon(want.want_id)
        else:
            self.counts["bid"] += 1
            want.bids += 1
        v = bind_verdict(decision, now, end)
        want.funded = bool(decision is not None and getattr(decision, "outcome", None) == "funded")
        if decision is not None and not want.funded:
            _release_soon(want.want_id)   # a held or pool-queued want reserves nothing
        why = str(v.detail.get("why") or "")
        if v.admit:
            self.counts["no_answer" if v.why.startswith("no_answer") else "admitted"] += 1
        else:
            self.counts["held"] += 1
            want.holds += 1
            want.retry_at = v.retry_at
            want.last_reason = str(v.detail.get("reason") or v.why)
            want.first_why = want.first_why or why
        want.last_why = why or want.last_why
        self._log(want.key, want.door, want, v)
        return v

    async def close(self, key: str, *, sent: bool, want_id: Optional[str] = None) -> None:
        """The door's attempt on an ADMITTED want ended: the next check opens a new want. A funded
        reservation is released now, sent or not (the act is over; a sent message is counted by
        ``sent_today`` from here), as the dream and run_bids do. A held want is left open (no-op).
        ``want_id``: close only that want (a check that did not admit it never closes another's)."""
        want = self.wants.get(key)
        if want is None or not want.admitted or (want_id is not None and want.want_id != want_id):
            return
        self.wants.pop(key, None)
        if want.funded:
            try:
                await asyncio.wait_for(_arelease(want.want_id), 2.0)
            except Exception:  # noqa: BLE001
                logger.info("physiology_bind_release_failed want_id=%s", want.want_id)

    def _forget(self, now: datetime) -> None:
        stale = [k for k, w in self.wants.items()
                 if not w.admitted and (now - (w.last_check_at or w.opened_at)).total_seconds() > FORGET_AFTER_SEC]
        for k in stale:
            self.wants.pop(k, None)

    def record(self, key: str, v: BindVerdict) -> Dict[str, Any]:
        """What the decision row carries (``result_json.economy``): the verdict and the market's own reason."""
        # A forced verdict owns no want: never borrow a held want's id (review finding 2).
        want = None if v.why == "forced" else self.wants.get(key)
        d = v.detail or {}
        out: Dict[str, Any] = {"mode": "on", "family": FAMILY, "producer": BINDING_PRODUCER, "verdict": v.why,
                               "admit": v.admit}
        for k in ("decision_id", "outcome", "reason", "binding_constraint", "lane", "net", "charge",
                  "retry_after_sec", "why"):
            if d.get(k) is not None:
                out[k] = d[k] if k != "why" else str(d[k])[:600]
        if v.retry_at is not None and not v.admit:
            out["retry_at"] = v.retry_at.isoformat()
        if want is not None:
            out.update(want_id=want.want_id, door=want.door, holds=want.holds, bids=want.bids)
            if want.first_bid_at is not None:
                out["first_bid_at"] = want.first_bid_at.isoformat()
                out["held_sec"] = round(max(0.0, ((want.admitted_at or want.last_check_at or want.first_bid_at)
                                                  - want.first_bid_at).total_seconds()), 1)
            if want.first_why and want.holds:
                out["first_why"] = want.first_why[:300]
        return out

    def status(self) -> Dict[str, Any]:
        return {"counts": dict(self.counts),
                "open": [{"key": w.key, "door": w.door, "want_id": w.want_id, "holds": w.holds, "admitted": w.admitted,
                          "retry_at": w.retry_at.isoformat() if w.retry_at else None, "last_reason": w.last_reason}
                         for w in list(self.wants.values())[:20]]}

    @staticmethod
    def _log(key: str, door: str, want: Optional[HeldWant], v: BindVerdict) -> None:
        d = v.detail or {}
        logger.info(
            "physiology_bind family=%s door=%s key=%s want_id=%s verdict=%s admit=%s holds=%d retry_at=%s "
            "decision_id=%s outcome=%s reason=%s binding=%s net=%s charge=%s why=%r obeyed=true",
            FAMILY, door, key, want.want_id if want else "-", v.why, v.admit, want.holds if want else 0,
            v.retry_at.isoformat() if (v.retry_at and not v.admit) else None, d.get("decision_id"),
            d.get("outcome"), d.get("reason"), d.get("binding_constraint"), d.get("net"), d.get("charge"),
            str(d.get("why") or "")[:300],
        )


__all__ = ["ADMITTED_STALE_SEC", "BINDING_PRODUCER", "DEFERRABLE_WINDOW_S", "DUPLICATE_REASON", "FAMILY",
           "HELD_REASON", "HeldWant", "OutreachBinder", "TICK_KEY", "bind_mode"]
