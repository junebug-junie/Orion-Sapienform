"""Curiosity posts an event want for each new philosophical finding (curiosity event wants, step 3).

Design: docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md ("Which findings compete"),
in the pattern of ``scripts/reading_event_wants.py``. Step 2 (``scripts/curiosity_finding_register.py``)
writes one ``curiosity_finding_register`` row per finished curiosity turn: the judge's register
(philosophical / operational / no_finding) and the finding's novelty against everything Orion wrote before.

The event: one register row with ``register = 'philosophical'`` and a non-null ``novelty``. Its magnitude
is that novelty. Its salience is the novelty as the source's own strict-below percentile against the
philosophical findings of the previous 30 days (``orion.physiology.event_wants.source_salience``; under 20
past findings: unknown, nothing is posted, never retried).

The band: only a finding with salience >= ``SALIENCE_BAND`` (0.9, the top tenth) is posted. Novelty is a
near-continuous number, so its percentile almost never reads 0 (the rest point reading reaches through ties
at the bottom): without the band about half of all philosophical findings would read above 0.5 and every
finding would compete. A finding under the band is logged as skipped and never retried (its history only
grows).

The want: family ``hub.curiosity_finding`` (Tier 2, discretionary, like the curiosity run it comes from),
verb ``infer``: the act a funded want leads to is Prime thinking the idea further (Prime's thoughts are
``infer`` acts in the act ledger), not a perception (``look``, reading's verb: a read's links are things in
the world to look at). ``event_ref`` ``curiosity_run:<run_id>``, ``opened_at`` the register row's
``created_at`` (when the finding was scored). One want per finding: the want id is deterministic, and a
finding whose want is already in ``want_ledger`` is not posted again (a restart re-reads the ledger).

Cost: no GPU slot (``orion.physiology.appetite.no_slot_cost``). The finding's model time is already spent
and charged to the curiosity run's act; the thought it leads to is Prime's, and Prime bids for its own lane.

SHADOW: the decision is logged and nothing in Hub obeys it. Prime reads the funded ones as concerns
(orion-mind ``PRIME_CONCERNS_FROM_EVENT_WANTS``), labelled with the finding's own words by
orion-durable-runs ``app/event_wants_menu.py``.

Known risk: the register's judge (prompt v2, Flash-Next) has not passed the dressing eval (#3435, 17.6% of
operational findings flipped to philosophical when rewritten reflectively). A dressed plumbing finding can
compete here. Juniper chose to ship (2026-11-16) rather than tune first.

Flag: ``HUB_CURIOSITY_EVENT_WANTS_ENABLED`` (default true). Off: no read, no bid.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from orion.physiology.appetite import no_slot_cost
from orion.physiology.event_wants import SOURCE_HISTORY, event_want, event_want_id, source_salience
from orion.schemas.grammar import ActCostV2

logger = logging.getLogger("orion-hub.curiosity_event_wants")

SOURCE = "curiosity_finding"
FAMILY = "hub.curiosity_finding"
VERB = "infer"
TIER = 2
PRODUCER = "orion-hub.curiosity_findings"
THOUGHT_COST = ActCostV2(**no_slot_cost())
EVENT_PREFIX = "curiosity_run"
UNAVAILABLE = "physiology_unavailable"
# Only the top tenth of philosophical findings by novelty competes (module docstring, "The band").
SALIENCE_BAND = 0.9

# Philosophical findings with a novelty, scored in [$1, $2).
FINDINGS_SQL = """
SELECT run_id, line, novelty, created_at
FROM curiosity_finding_register
WHERE register = 'philosophical' AND novelty IS NOT NULL AND created_at >= $1 AND created_at < $2
ORDER BY created_at
"""

POSTED_SQL = "SELECT DISTINCT want_id FROM want_ledger WHERE reason <> 'physiology_unavailable' AND want_id = ANY($1::text[]) AND decided_at >= $2"


def plan_findings(rows: list[Any], *, now: datetime, lookback: timedelta,
                  band: float = SALIENCE_BAND) -> list[dict[str, Any]]:
    """The philosophical findings scored inside ``lookback``, each with its salience against the
    findings of the 30 days before it (None: too little history) and whether it is in the band. Pure."""
    found = sorted(({"run_id": str(r["run_id"]), "line": str(r["line"] or ""), "at": r["created_at"],
                     "novelty": float(r["novelty"])}
                    for r in rows if r["created_at"] is not None and r["novelty"] is not None),
                   key=lambda r: r["at"])
    out = []
    for r in found:
        if not now - lookback <= r["at"] <= now:
            continue
        history = [h["novelty"] for h in found if r["at"] - SOURCE_HISTORY <= h["at"] < r["at"]]
        salience = source_salience(r["novelty"], history)
        out.append({**r, "salience": salience, "history_n": len(history),
                    "in_band": salience is not None and salience >= band})
    return out


class CuriosityEventWants:
    def __init__(self, *, enabled: bool, pool_provider: Callable[[], Any],
                 bid: Optional[Callable[[Any], Awaitable[Any]]] = None, tick_interval_sec: float = 300.0,
                 lookback_sec: float = 43200.0, band: float = SALIENCE_BAND,
                 clock: Optional[Callable[[], datetime]] = None) -> None:
        self.enabled = bool(enabled)
        self._pool_provider = pool_provider
        self._bid = bid
        self.tick_interval_sec = max(30.0, float(tick_interval_sec))
        self.lookback = timedelta(seconds=max(60.0, float(lookback_sec)))
        self.band = float(band)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # Process-local mirror of what was decided (posted, skipped) -- the ledger row may lag the bid.
        self._posted: set[str] = set()
        self._task: Optional[asyncio.Task] = None
        self.last_tick: dict[str, Any] = {}

    async def start(self) -> None:
        if not self.enabled:
            logger.info("curiosity_event_wants disabled")
            return
        self._task = asyncio.create_task(self._run())
        logger.info("curiosity_event_wants started tick=%ss lookback=%ss band=%.2f", self.tick_interval_sec,
                    self.lookback.total_seconds(), self.band)

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
                logger.exception("curiosity_event_wants_tick_failed")
            await asyncio.sleep(self.tick_interval_sec)

    async def _abid(self, want: Any) -> Any:
        if self._bid is not None:
            return await self._bid(want)
        from orion.physiology.client import abid

        return await abid(want)

    async def tick(self) -> dict[str, Any]:
        now = self._clock()
        summary: dict[str, Any] = {"at": now.isoformat(), "posted": 0, "no_history": 0, "below_band": 0,
                                   "already": 0}
        self.last_tick = summary
        pool = self._pool_provider()
        if pool is None:
            summary["blocked"] = "no_pool"
            return summary
        async with pool.acquire() as conn:
            rows = await conn.fetch(FINDINGS_SQL, now - self.lookback - SOURCE_HISTORY, now)
            plan = plan_findings(list(rows), now=now, lookback=self.lookback, band=self.band)
            ids = [event_want_id(SOURCE, p["run_id"]) for p in plan]
            fresh = [i for i, p in zip(ids, plan) if i not in self._posted and p["in_band"]]
            posted = ({str(r["want_id"]) for r in await conn.fetch(POSTED_SQL, fresh, now - self.lookback - timedelta(days=1))}
                      if fresh else set())
        self._posted |= posted
        for p, want_id in zip(plan, ids):
            if want_id in self._posted:
                summary["already"] += 1
                continue
            if p["salience"] is None:
                summary["no_history"] += 1
                logger.info("curiosity_event_want_skipped run=%s reason=no_history history_n=%d",
                            p["run_id"], p["history_n"])
                self._posted.add(want_id)     # the history only grows past this finding: never retried
                continue
            if not p["in_band"]:
                summary["below_band"] += 1
                logger.info("curiosity_event_want_skipped run=%s reason=below_band novelty=%.4f salience=%.3f "
                            "band=%.2f history_n=%d", p["run_id"], p["novelty"], p["salience"], self.band,
                            p["history_n"])
                self._posted.add(want_id)     # its salience is fixed once scored: never retried
                continue
            want = event_want(source=SOURCE, key=p["run_id"], family=FAMILY, producer=PRODUCER, verb=VERB,
                              tier=TIER, event_ref=f"{EVENT_PREFIX}:{p['run_id']}", salience=p["salience"],
                              opened_at=p["at"], cost=THOUGHT_COST)
            decision = await self._abid(want)
            if decision is None or decision.reason == UNAVAILABLE:
                # No answer, or the market could not price it: not a decision about the finding, so it is
                # bid again next tick while the finding is in its window.
                logger.info("curiosity_event_want_unanswered run=%s reason=%s", p["run_id"],
                            getattr(decision, "reason", None) or "no_answer")
                continue
            self._posted.add(want_id)
            summary["posted"] += 1
            logger.info("curiosity_event_want run=%s line=%s novelty=%.4f salience=%.3f outcome=%s reason=%s "
                        "value_rank=%s want_id=%s", p["run_id"], p["line"], p["novelty"], p["salience"],
                        decision.outcome, decision.reason, decision.value_rank, want_id)
        if len(self._posted) > 5000:
            self._posted = set(ids)   # a memory guard; the ledger still answers for older findings
        return summary


__all__ = ["FAMILY", "FINDINGS_SQL", "SALIENCE_BAND", "CuriosityEventWants", "plan_findings"]
