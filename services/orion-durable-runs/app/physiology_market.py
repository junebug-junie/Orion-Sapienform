"""The bid market's reads and state (body economy Patch C1: Appetite + Choice, SHADOW).

``POST /physiology/bid`` (app.main) calls ``Market.bid``. Per bid:

1. the latest ``PhysiologyStateV1`` from Redis ``orion:physiology:latest`` (the posted prices);
2. the act history: ``grammar_acts`` for the last 30 days plus each actor's newest close, and
   ``act_downstream_use`` for the same window -- cached for ``ORION_PHYSIOLOGY_BID_HISTORY_TTL_SEC``
   (default 300 s; weights move at most 0.1 per day, appetite by minutes);
3. the pure decision (``orion.physiology.choice.decide``) against the in-memory reservation book;
4. the decision published on ``orion:physiology:want:decision`` (sql-writer -> ``want_ledger``).

The exploration floor's weekly quota (Patch C3a) counts funded wants from ``want_ledger`` over the
rolling 7 days, re-read with the act history; a failed ledger read leaves the last count (and no
floor until a first read succeeds), it never fails the bid.

Bid sites in this process (Patch C3a): durable-runs' own admission (``app.run_bids``) calls
``Market.bid`` in-process, ``mark_opened`` at the run's first grant and ``release`` at its terminal.

Prime's regulate signals (Orion Prime step 3, ``orion.physiology.prime_charge``). When
``prime_regulation`` is on and the want's family declares the signal's direction as a receptor
(config/physiology/families.yaml), each bid reads ``orion:prime:regulate:<family>`` from the same Redis
as the prices. A TREATED signal in force is applied as a bounded charge (slow) or credit (quicken) in
``decide``; a CONTROL signal (Prime's holdback) is not applied, but what it would have done is computed.
Either way the market writes what the charge did to this want back into the same hash
(``w:<want_id>``), which is how Prime grades its signal and sees "did curiosity outbid me, and why".
Off, unreadable, expired or not received: no charge, exactly as before. A Redis error never fails a bid.
Only bids from ``orion.physiology.binding.BINDING_PRODUCERS`` (the obeyed ones) read the signal: a shadow bid of
the same family (its producer rolled back to shadow, or a forced dream sleep) is never charged or recorded.

Read-only on Postgres. A failed history read is ``physiology_unavailable`` for that bid, never a
fund. SHADOW: the decision is returned and logged; nobody obeys it yet.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from orion.physiology.appetite import ActRecord
from orion.physiology.choice import (
    FLOOR_WINDOW,
    FundedWant,
    MarketContext,
    ReservationBook,
    SYNTHETIC_PRODUCERS,
    build_context,
    decide,
    unavailable,
)
from orion.physiology import prime_charge as pc
from orion.physiology.binding import BINDING_PRODUCERS
from orion.physiology.families import FamiliesConfig
from orion.physiology.weights import UseRecord, weights_now
from orion.schemas.physiology import PhysiologyStateV1, parse_physiology_state
from orion.schemas.physiology_want import WantDecisionV1, WantV1

logger = logging.getLogger("orion-durable-runs.physiology_market")

HISTORY_WINDOW = timedelta(days=30)

ACTS_SQL = """
SELECT act_id, actor, opened_at, closed_at, outcome_value, outcome_source, signal_id, cost_json
  FROM grammar_acts
 WHERE coalesce(opened_at, closed_at, created_at) >= %(since)s
UNION
(SELECT DISTINCT ON (actor) act_id, actor, opened_at, closed_at, outcome_value, outcome_source, signal_id, cost_json
   FROM grammar_acts
  WHERE closed_at IS NOT NULL AND closed_at < %(since)s
  ORDER BY actor, closed_at DESC)
"""

USES_SQL = """
SELECT act_id, actor, started_at, link_status, used, window_closed
  FROM act_downstream_use
 WHERE started_at >= %(since)s
"""


FUNDED_SQL = """
SELECT decision_id, family, tier, decided_at
  FROM want_ledger
 WHERE outcome = 'funded' AND decided_at >= %(since)s AND NOT (producer = ANY(%(synthetic)s))
"""


def _utc(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


def act_from_row(r: Any) -> ActRecord:
    cost = r.get("cost_json")
    if isinstance(cost, str):
        try:
            cost = json.loads(cost)
        except ValueError:
            cost = {}
    return ActRecord(act_id=str(r["act_id"]), family=str(r["actor"]), opened_at=_utc(r.get("opened_at")),
                     closed_at=_utc(r.get("closed_at")),
                     outcome_value=float(r["outcome_value"]) if r.get("outcome_value") is not None else None,
                     outcome_source=r.get("outcome_source"), signal_id=r.get("signal_id"),
                     cost=cost if isinstance(cost, dict) else {})


def use_from_row(r: Any) -> UseRecord:
    return UseRecord(act_id=str(r["act_id"]), family=str(r["actor"]), started_at=_utc(r.get("started_at")),
                     link_status=str(r["link_status"]), used=r.get("used"), window_closed=bool(r.get("window_closed")))


async def read_history(pool: Any, now: datetime) -> tuple[list[ActRecord], list[UseRecord]]:
    """Raises on a failed read (the caller turns it into physiology_unavailable)."""
    since = now - HISTORY_WINDOW
    async with pool.connection() as conn:
        acts = [act_from_row(r) for r in await (await conn.execute(ACTS_SQL, {"since": since})).fetchall()]
        uses = [use_from_row(r) for r in await (await conn.execute(USES_SQL, {"since": since})).fetchall()]
    return acts, uses


async def read_funded(pool: Any, now: datetime) -> list[FundedWant]:
    """want_ledger's funded wants over the floor's rolling week. Raises on a failed read."""
    async with pool.connection() as conn:
        rows = await (await conn.execute(FUNDED_SQL, {"since": now - FLOOR_WINDOW,
                                                      "synthetic": sorted(SYNTHETIC_PRODUCERS)})).fetchall()
    return [FundedWant(str(r["decision_id"]), _utc(r["decided_at"]), str(r["family"]), int(r["tier"])) for r in rows]


@dataclass
class _Cached:
    at: datetime
    acts: list[ActRecord]
    uses: list[UseRecord]


class Market:
    """One per durable-runs process: the families config, the history cache, the reservation book."""

    def __init__(self, *, families: Optional[FamiliesConfig], families_error: Optional[str], pool_getter,
                 redis_getter, publish, history_ttl_sec: float = 300.0, lane_window_sec: float = 3600.0,
                 stale_after_sec: float = 360.0, prime_regulation: bool = False) -> None:
        self.families = families
        self.families_error = families_error
        self._pool_getter = pool_getter
        self._redis_getter = redis_getter
        self._publish = publish
        self._ttl = history_ttl_sec
        self._lane_window = lane_window_sec
        self._stale = stale_after_sec
        self._cache: Optional[_Cached] = None
        # Weights are a 30-day daily replay: computed once per history snapshot and UTC day.
        self._weights: Optional[tuple[tuple[datetime, Any], dict]] = None
        self._lock = asyncio.Lock()
        self.book = ReservationBook()
        # Orion Prime step 3: apply Prime's regulate charges (ORION_PHYSIOLOGY_PRIME_REGULATION). Off: the
        # signal is never read; bids are exactly as before.
        self.prime_regulation = bool(prime_regulation)
        self.counts = {"funded": 0, "deferred": 0, "publish_failed": 0, "prime_applied": 0, "prime_control": 0,
                       "prime_read_failed": 0}

    async def _history(self, now: datetime) -> _Cached:
        async with self._lock:
            c = self._cache
            if c is not None and 0 <= (now - c.at).total_seconds() <= self._ttl:
                return c
            acts, uses = await read_history(self._pool_getter(), now)
            if not acts:
                # An act ledger read that returns nothing is not "no hunger anywhere": unknown.
                raise RuntimeError("grammar_acts returned no rows")
            self._cache = _Cached(at=now, acts=acts, uses=uses)
            try:
                self.book.seed_funded(await read_funded(self._pool_getter(), now), now)
            except Exception:  # noqa: BLE001 -- keep the last count; no floor until a first read succeeds
                logger.warning("physiology_bid_funded_read_failed", exc_info=True)
            return self._cache

    async def _state(self) -> Optional[PhysiologyStateV1]:
        from orion.schemas.physiology import PHYSIOLOGY_STATE_REDIS_KEY

        redis = self._redis_getter()
        if redis is None:
            return None
        try:
            return parse_physiology_state(await redis.get(PHYSIOLOGY_STATE_REDIS_KEY))
        except Exception:  # noqa: BLE001 -- unread is unknown
            logger.warning("physiology_bid_state_read_failed", exc_info=True)
            return None

    async def context(self, state: Optional[PhysiologyStateV1], now: datetime) -> Optional[MarketContext]:
        if self.families is None:
            return None
        try:
            hist = await self._history(now)
        except Exception:  # noqa: BLE001
            logger.warning("physiology_bid_history_read_failed", exc_info=True)
            return None
        key = (hist.at, now.astimezone(timezone.utc).date())
        if self._weights is None or self._weights[0] != key:
            self._weights = (key, weights_now(self.families, hist.acts, hist.uses, now))
        ctx = build_context(self.families, hist.acts, hist.uses, state, now, lane_window_sec=self._lane_window,
                            weights=self._weights[1])
        # dataclasses.replace keeps every field (C1 rebuilt the context by hand and dropped
        # tier1_recent_use, so Tier 2's line forgot Tier 1's measured use).
        return dataclasses.replace(ctx, stale_after_sec=self._stale)

    async def bid(self, want: WantV1, now: Optional[datetime] = None, *, reserve_at_open: bool = False) -> WantDecisionV1:
        now = now or datetime.now(timezone.utc)
        state = await self._state()
        ctx = await self.context(state, now)
        if ctx is None:
            why = (f"families config unavailable ({self.families_error})" if self.families is None
                   else "act history unread")
            # Never fund on a missing market: physiology_unavailable.
            decision = unavailable(want, now, why=why, state=state)
        else:
            # Only an OBEYED bid takes Prime's charge (a shadow bid would grade Prime on holds that never happened).
            sig = await self._prime_signal(want.family, now) if want.producer in BINDING_PRODUCERS else None
            applied = sig if sig is not None and sig.arm == "treated" else None
            other = None
            if sig is not None:
                # The same bid decided the other way, on a throwaway copy of the book (it must not reserve or
                # count anything): what the charge changed is read from two real decisions, both arms alike.
                other = decide(want, state, ctx, copy.deepcopy(self.book), now, reserve_at_open=reserve_at_open,
                               prime=None if applied is not None else sig)
            decision = decide(want, state, ctx, self.book, now, reserve_at_open=reserve_at_open, prime=applied)
            if sig is not None and other is not None:
                await self._record_prime_effect(sig, want, decision, other, now, applied=applied is not None)
        self.counts[decision.outcome] = self.counts.get(decision.outcome, 0) + 1
        try:
            await self._publish(decision)
        except Exception:  # noqa: BLE001 -- the caller still gets its decision; the ledger row is lost
            self.counts["publish_failed"] += 1
            logger.warning("physiology_bid_publish_failed want=%s", want.want_id, exc_info=True)
        logger.info("physiology_bid family=%s outcome=%s reason=%s lane=%s net=%s binding=%s shadow=true",
                    decision.family, decision.outcome, decision.reason, decision.lane, decision.net,
                    decision.binding_constraint)
        return decision

    async def _prime_signal(self, family: str, now: datetime) -> Optional[pc.PrimeRegulationV1]:
        """The Prime signal in force for ``family``, or None (flag off, no receptor, absent, expired,
        unreadable). Never raises: an unread signal is no charge."""
        if not self.prime_regulation or self.families is None:
            return None
        cfg = self.families.get(family)
        if cfg is None or not cfg.receptors:
            return None
        redis = self._redis_getter()
        if redis is None:
            return None
        try:
            raw = await redis.hget(pc.key(family), pc.SIGNAL_FIELD)
        except Exception:  # noqa: BLE001
            self.counts["prime_read_failed"] += 1
            logger.warning("prime_charge_read_failed family=%s", family, exc_info=True)
            return None
        sig = pc.parse_signal(raw)
        if sig is None or sig.family != family or not pc.active(sig, now):
            return None
        if not pc.receives(cfg.receptors, sig.direction):
            logger.warning("prime_charge_not_received family=%s direction=%s receptors=%s signal=%s",
                           family, sig.direction, list(cfg.receptors), sig.signal_id)
            return None
        return sig

    async def _record_prime_effect(self, sig: pc.PrimeRegulationV1, want: WantV1, decision: WantDecisionV1,
                                   other: WantDecisionV1, now: datetime, *, applied: bool) -> None:
        """Write what the charge did (treated) or would have done (control) to this want. Best-effort.
        ``decision`` is the one the market made; ``other`` the same bid decided the other way."""
        if decision.net is None or other.net is None:
            return          # the market could not price the want: nothing to say about the charge
        with_c, without_c = (decision, other) if applied else (other, decision)
        eff = pc.effect_between(sig.direction, with_c, without_c)
        self.counts["prime_applied" if applied else "prime_control"] += 1
        why = decision.why if applied else f"held back (control, not applied): with it, {with_c.why}"
        net_with, net_without = with_c.net, without_c.net
        redis = self._redis_getter()
        if redis is None:
            return
        k, field_name = pc.key(sig.family), f"{pc.WANT_FIELD_PREFIX}{want.want_id}"
        try:
            prev = await redis.hget(k, field_name)
            if prev is None and await redis.hlen(k) >= pc.MAX_WANT_FIELDS + 1:      # + the signal field
                return
            prev_m = None
            if prev is not None:
                try:
                    prev_m = pc.PrimeChargeEffectV1.model_validate_json(prev.decode() if isinstance(prev, bytes) else prev)
                except Exception:  # noqa: BLE001
                    prev_m = None
            rec = pc.merge(prev_m, want.want_id, eff, now, why, signal_id=sig.signal_id)
            await redis.hset(k, field_name, pc.dumps(rec))
            # Never leave a key without an expiry (hset on a key that just expired would recreate it).
            await redis.expire(k, max(1, int((sig.expires_at - now).total_seconds() + pc.GRACE_SEC)))
        except Exception:  # noqa: BLE001
            logger.warning("prime_charge_record_failed family=%s want=%s", sig.family, want.want_id, exc_info=True)
            return
        logger.info("prime_charge family=%s want=%s arm=%s direction=%s effect=%s net_with=%.3f net_without=%.3f "
                    "signal=%s act=%s", sig.family, want.want_id, sig.arm, sig.direction, eff, net_with, net_without,
                    sig.signal_id, sig.act_id)

    def release(self, want_id: str) -> int:
        return self.book.release(want_id)

    def mark_opened(self, want_id: str, at: Optional[datetime] = None) -> int:
        return self.book.mark_opened(want_id, at or datetime.now(timezone.utc))

    async def family_weights(self, now: Optional[datetime] = None) -> dict:
        """Today's learned family weights with their evidence (the same ones bids use), for the
        read-only report. Raises when the families config or the act history is unavailable."""
        now = now or datetime.now(timezone.utc)
        if self.families is None:
            raise RuntimeError(f"families config unavailable ({self.families_error})")
        hist = await self._history(now)
        key = (hist.at, now.astimezone(timezone.utc).date())
        if self._weights is None or self._weights[0] != key:
            self._weights = (key, weights_now(self.families, hist.acts, hist.uses, now))
        return dict(self._weights[1])

    def health(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now(timezone.utc)
        return {"families": len(self.families.families) if self.families else 0,
                "families_error": self.families_error, "history_cached_at": self._cache.at.isoformat() if self._cache else None,
                "reservations": len(self.book.snapshot(now)), "funded_seeded": self.book.funded_seeded,
                "prime_regulation": self.prime_regulation, **self.counts}


__all__ = ["Market", "act_from_row", "read_funded", "read_history", "use_from_row"]
