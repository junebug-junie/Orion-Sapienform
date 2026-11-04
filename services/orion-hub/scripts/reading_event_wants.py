"""Reading posts an event want for each finished read (attention as the want competition, patch 2).

Design: docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md, "1. The menu: event
wants" (v1 source: reading findings). The event is one finished read (``world_pulse_read_seed`` with
``stage2_status = 'done'``). Its magnitude is how many links the read made: the Stage 2 relationship
claims whose receipt is ``accepted_provisional`` (the ``link_made`` finding of
``orion/world_pulse_read/findings.py``, counted from the same receipts). Its salience is that count as
the source's own strict-below percentile against the reads of the previous 30 days
(``orion.physiology.event_wants.source_salience``): a read that made no link reads 0, the rest point.

The want: family ``hub.world_pulse_read`` (Tier 2), verb ``look`` (a read that changed something is
worth a follow-up look), ``event_ref`` ``world_pulse_read_seed:<seed_id>``, ``opened_at`` the read's
Stage 2 completion. Its window is the family's (3 h from the read). One want per read: the want id is
deterministic, and a read whose want is already in ``want_ledger`` is not posted again (a restart
re-reads the ledger).

SHADOW: the decision is logged and nothing in Hub obeys it. Prime reads the funded ones as its concerns
(orion-mind ``PRIME_CONCERNS_FROM_EVENT_WANTS``). Belief moves (a curiosity run that later cites the
read) are not counted: they land days after the read, past this want's window.

Flag: ``HUB_READING_EVENT_WANTS_ENABLED`` (default true). Off: no read, no bid.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from orion.physiology.event_wants import SOURCE_HISTORY, event_want, event_want_id, source_salience

logger = logging.getLogger("orion-hub.reading_event_wants")

SOURCE = "reading"
FAMILY = "hub.world_pulse_read"
VERB = "look"
TIER = 2
PRODUCER = "orion-hub.reading_events"
EVENT_PREFIX = "world_pulse_read_seed"
UNAVAILABLE = "physiology_unavailable"

# Finished reads in [$1, $2) with the links each made (accepted relationship claims).
READS_SQL = """
SELECT s.seed_id, s.url, s.stage2_completed_at,
       (SELECT count(*) FROM jsonb_array_elements(
            CASE WHEN jsonb_typeof(s.stage2_result_json->'relationship_claims') = 'array'
                 THEN s.stage2_result_json->'relationship_claims' ELSE '[]'::jsonb END) AS c(claim)
         WHERE c.claim->'receipt'->>'outcome' = 'accepted_provisional') AS links
FROM world_pulse_read_seed s
WHERE s.stage2_status = 'done' AND s.stage2_completed_at >= $1 AND s.stage2_completed_at < $2
ORDER BY s.stage2_completed_at
"""

POSTED_SQL = "SELECT DISTINCT want_id FROM want_ledger WHERE reason <> 'physiology_unavailable' AND want_id = ANY($1::text[]) AND decided_at >= $2"


def plan_reads(rows: list[Any], *, now: datetime, lookback: timedelta) -> list[dict[str, Any]]:
    """The reads finished inside ``lookback`` with their salience (None: too little history). Pure.
    Each read is ranked against the reads finished in the 30 days before it."""
    reads = sorted(({"seed_id": str(r["seed_id"]), "url": str(r["url"] or ""), "at": r["stage2_completed_at"],
                     "links": int(r["links"] or 0)} for r in rows if r["stage2_completed_at"] is not None),
                   key=lambda r: r["at"])
    out = []
    for r in reads:
        if not now - lookback <= r["at"] <= now:
            continue
        history = [h["links"] for h in reads if r["at"] - SOURCE_HISTORY <= h["at"] < r["at"]]
        out.append({**r, "salience": source_salience(r["links"], history), "history_n": len(history)})
    return out


class ReadingEventWants:
    def __init__(self, *, enabled: bool, pool_provider: Callable[[], Any],
                 bid: Optional[Callable[[Any], Awaitable[Any]]] = None, tick_interval_sec: float = 300.0,
                 lookback_sec: float = 10800.0, clock: Optional[Callable[[], datetime]] = None) -> None:
        self.enabled = bool(enabled)
        self._pool_provider = pool_provider
        self._bid = bid
        self.tick_interval_sec = max(30.0, float(tick_interval_sec))
        self.lookback = timedelta(seconds=max(60.0, float(lookback_sec)))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # Process-local mirror of what was posted (the ledger row may lag the bid).
        self._posted: set[str] = set()
        self._task: Optional[asyncio.Task] = None
        self.last_tick: dict[str, Any] = {}

    async def start(self) -> None:
        if not self.enabled:
            logger.info("reading_event_wants disabled")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("reading_event_wants started tick=%ss lookback=%ss", self.tick_interval_sec,
                    self.lookback.total_seconds())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- one bad tick must not stop the loop
                logger.exception("reading_event_wants_tick_failed")
            await asyncio.sleep(self.tick_interval_sec)

    async def _abid(self, want: Any) -> Any:
        if self._bid is not None:
            return await self._bid(want)
        from orion.physiology.client import abid

        return await abid(want)

    async def tick(self) -> dict[str, Any]:
        now = self._clock()
        summary: dict[str, Any] = {"at": now.isoformat(), "posted": 0, "no_history": 0, "already": 0}
        self.last_tick = summary
        pool = self._pool_provider()
        if pool is None:
            summary["blocked"] = "no_pool"
            return summary
        async with pool.acquire() as conn:
            rows = await conn.fetch(READS_SQL, now - self.lookback - SOURCE_HISTORY, now)
            plan = plan_reads(list(rows), now=now, lookback=self.lookback)
            ids = [event_want_id(SOURCE, p["seed_id"]) for p in plan]
            fresh = [i for i in ids if i not in self._posted]
            posted = ({str(r["want_id"]) for r in await conn.fetch(POSTED_SQL, fresh, now - self.lookback - timedelta(days=1))}
                      if fresh else set())
        self._posted |= posted
        for p, want_id in zip(plan, ids):
            if want_id in self._posted:
                summary["already"] += 1
                continue
            if p["salience"] is None:
                summary["no_history"] += 1
                logger.info("reading_event_want_skipped seed=%s reason=no_history history_n=%d",
                            p["seed_id"], p["history_n"])
                self._posted.add(want_id)     # the history only grows past this read: never retried
                continue
            want = event_want(source=SOURCE, key=p["seed_id"], family=FAMILY, producer=PRODUCER, verb=VERB,
                              tier=TIER, event_ref=f"{EVENT_PREFIX}:{p['seed_id']}", salience=p["salience"],
                              opened_at=p["at"], object_ref=p["url"] or None)
            decision = await self._abid(want)
            if decision is None or decision.reason == UNAVAILABLE:
                # No answer, or the market could not price it (no fresh body, history unread): not a
                # decision about the event, so it is bid again next tick while the read is in its window.
                logger.info("reading_event_want_unanswered seed=%s reason=%s", p["seed_id"],
                            getattr(decision, "reason", None) or "no_answer")
                continue
            self._posted.add(want_id)
            summary["posted"] += 1
            logger.info("reading_event_want seed=%s links=%d salience=%.3f outcome=%s reason=%s value_rank=%s want_id=%s",
                        p["seed_id"], p["links"], p["salience"], decision.outcome, decision.reason,
                        decision.value_rank, want_id)
        if len(self._posted) > 5000:
            self._posted = set(ids)   # a memory guard; the ledger still answers for older reads
        return summary


__all__ = ["ReadingEventWants", "plan_reads"]
