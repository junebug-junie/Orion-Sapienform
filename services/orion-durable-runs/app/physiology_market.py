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

Read-only on Postgres. A failed history read is ``physiology_unavailable`` for that bid, never a
fund. SHADOW: the decision is returned and logged; nobody obeys it yet.
"""

from __future__ import annotations

import asyncio
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
                 stale_after_sec: float = 360.0) -> None:
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
        self.counts = {"funded": 0, "deferred": 0, "publish_failed": 0}

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
            decision = decide(want, state, ctx, self.book, now, reserve_at_open=reserve_at_open)
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
                "reservations": len(self.book.snapshot(now)), "funded_seeded": self.book.funded_seeded, **self.counts}


__all__ = ["Market", "act_from_row", "read_funded", "read_history", "use_from_row"]
