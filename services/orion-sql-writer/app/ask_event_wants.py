"""Camera asks post an event want each (attention as the want competition, patch 2).

Design: docs/superpowers/specs/2026-11-04-attention-as-want-competition-design.md, "1. The menu: event
wants" (v1 source: camera arrivals and asks). The event is one OPEN camera ask in ``orion_ask``
(``vision_individual`` "whose vehicle is this?", ``vision_person_confirm`` "is this Jacob?",
``vision_same_individual`` "are these the same?"; opened by app/vision_individuals.py and
app/vision_handoff.py). An ask answered before this loop saw it open was never an open event and is
not posted.

Magnitude: how many times the camera had seen the individual the ask is about (the first individual in
``source_ref``) when the ask opened, counted from ``vision_individual_sighting`` (moving sightings
only, as the asks themselves count). Salience: that count as the source's own strict-below percentile
against every camera ask of the previous 30 days, each counted the same way at its own open
(``orion.physiology.event_wants.source_salience``). Why this magnitude: a name for an individual seen
many times resolves many past and future sightings; one seen twice resolves two.

The want: family ``vision.ask`` (Tier 2, a proposal awaiting Juniper's sign-off), verb ``ask``,
``event_ref`` ``orion_ask:<ask_id>``, ``object_ref`` ``vision_individual:<id>``, ``opened_at`` the ask's
open, window end the ask's ``expires_at``. One want per ask (deterministic want id; an ask whose want
is already in ``want_ledger`` is not posted again).

SHADOW: nothing obeys the decision. Prime reads the funded ones as its concerns (orion-mind
``PRIME_CONCERNS_FROM_EVENT_WANTS``).

Env: ``ASK_EVENT_WANTS_INTERVAL_SEC`` (default 300; 0 disables), ``ASK_EVENT_WANTS_LOOKBACK_HOURS``
(default 24: an open ask older than this is not posted).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Sequence

from orion.physiology.event_wants import SOURCE_HISTORY, event_want, event_want_id, source_salience

logger = logging.getLogger("sql-writer.ask_event_wants")

SOURCE = "camera_ask"
FAMILY = "vision.ask"
VERB = "ask"
TIER = 2
PRODUCER = "orion-sql-writer.ask_events"
UNAVAILABLE = "physiology_unavailable"
CAMERA_ASK_KINDS = ("vision_individual", "vision_person_confirm", "vision_same_individual")

# Camera asks opened in [:since, :until), each with the moving sightings of its individual up to its open.
ASKS_SQL = """
WITH a AS (
    SELECT ask_id, source_kind, status, created_at, expires_at, split_part(source_ref, '|', 1) AS iid
      FROM orion_ask
     WHERE source_kind = ANY(:kinds) AND created_at >= :since AND created_at < :until)
SELECT a.ask_id, a.source_kind, a.status, a.created_at, a.expires_at, a.iid,
       (SELECT count(*) FROM vision_individual_sighting v
         WHERE v.individual_id = a.iid AND v.started_at <= a.created_at AND NOT v.stationary) AS sightings
  FROM a ORDER BY a.created_at
"""

POSTED_SQL = "SELECT DISTINCT want_id FROM want_ledger WHERE reason <> 'physiology_unavailable' AND want_id = ANY(:ids) AND decided_at >= :since"


def _utc(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


def plan_asks(rows: Sequence[Any], *, now: datetime, lookback: timedelta) -> list[dict[str, Any]]:
    """The OPEN camera asks opened inside ``lookback``, each with its salience (None: too little
    history). Each ask is ranked against the camera asks of the 30 days before it. Pure."""
    asks = sorted(({"ask_id": str(r["ask_id"]), "kind": str(r["source_kind"]), "status": str(r["status"]),
                    "at": _utc(r["created_at"]), "expires_at": _utc(r["expires_at"]), "iid": str(r["iid"] or ""),
                    "sightings": int(r["sightings"] or 0)} for r in rows if r["created_at"] is not None),
                  key=lambda a: a["at"])
    out = []
    for a in asks:
        if a["status"] != "open" or a["at"] < now - lookback or a["at"] > now:
            continue
        if a["expires_at"] is not None and a["expires_at"] <= now:
            continue
        history = [h["sightings"] for h in asks if a["at"] - SOURCE_HISTORY <= h["at"] < a["at"]]
        out.append({**a, "salience": source_salience(a["sightings"], history), "history_n": len(history)})
    return out


def run_one_cycle(*, postgres_uri: str, lookback: timedelta, posted: set[str], bid: Callable[[Any], Any],
                  now: Optional[datetime] = None, engine: Any = None) -> dict[str, Any]:
    """Read, plan and bid once (sync: runs in a worker thread). ``posted`` is updated in place."""
    from sqlalchemy import create_engine, text

    ts = now or datetime.now(timezone.utc)
    eng = engine or create_engine(postgres_uri, pool_pre_ping=True)
    summary: dict[str, Any] = {"posted": 0, "no_history": 0, "already": 0, "unanswered": 0}
    try:
        with eng.connect() as conn:
            rows = conn.execute(text(ASKS_SQL), {"kinds": list(CAMERA_ASK_KINDS),
                                                  "since": ts - lookback - SOURCE_HISTORY, "until": ts}).mappings().all()
            plan = plan_asks(rows, now=ts, lookback=lookback)
            ids = [event_want_id(SOURCE, p["ask_id"]) for p in plan]
            fresh = [i for i in ids if i not in posted]
            if fresh:
                posted |= {str(r[0]) for r in conn.execute(text(POSTED_SQL), {"ids": fresh,
                                                                              "since": ts - lookback - timedelta(days=1)})}
    finally:
        if engine is None:
            eng.dispose()
    for p, want_id in zip(plan, ids):
        if want_id in posted:
            summary["already"] += 1
            continue
        if p["salience"] is None:
            summary["no_history"] += 1
            logger.info("ask_event_want_skipped ask=%s reason=no_history history_n=%d", p["ask_id"], p["history_n"])
            posted.add(want_id)
            continue
        want = event_want(source=SOURCE, key=p["ask_id"], family=FAMILY, producer=PRODUCER, verb=VERB, tier=TIER,
                          event_ref=f"orion_ask:{p['ask_id']}", salience=p["salience"], opened_at=p["at"],
                          latest=p["expires_at"], object_ref=f"vision_individual:{p['iid']}" if p["iid"] else None)
        decision = bid(want)
        if decision is None or decision.reason == UNAVAILABLE:
            summary["unanswered"] += 1       # no answer, or the market could not price it: retried next cycle
            continue
        posted.add(want_id)
        summary["posted"] += 1
        logger.info("ask_event_want ask=%s kind=%s sightings=%d salience=%.3f outcome=%s reason=%s value_rank=%s want_id=%s",
                    p["ask_id"], p["kind"], p["sightings"], p["salience"], decision.outcome, decision.reason,
                    decision.value_rank, want_id)
    if len(posted) > 5000:
        posted.intersection_update(ids)
    return summary


async def ask_event_wants_loop(settings: Any) -> None:
    interval = float(getattr(settings, "ask_event_wants_interval_sec", 0.0) or 0.0)
    postgres_uri = str(getattr(settings, "postgres_uri", "") or "")
    if interval <= 0 or not postgres_uri:
        logger.info("ask_event_wants_loop DISABLED (ASK_EVENT_WANTS_INTERVAL_SEC=%s)", interval)
        return
    from sqlalchemy import create_engine

    from orion.physiology.client import bid

    lookback = timedelta(hours=float(getattr(settings, "ask_event_wants_lookback_hours", 24.0)))
    engine = create_engine(postgres_uri, pool_pre_ping=True, pool_size=1, max_overflow=0)
    posted: set[str] = set()
    logger.info("ask_event_wants_loop starting interval_sec=%.0f lookback_h=%.1f", interval,
                lookback.total_seconds() / 3600)
    while True:
        try:
            summary = await asyncio.to_thread(run_one_cycle, postgres_uri=postgres_uri, lookback=lookback,
                                              posted=posted, bid=bid, engine=engine)
            if summary["posted"] or summary["no_history"] or summary["unanswered"]:
                logger.info("[ASK_EVENT_WANTS] tick %s", summary)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one bad cycle must not stop the loop
            logger.warning("ask_event_wants_cycle_failed error=%s", exc)
        await asyncio.sleep(interval)


__all__ = ["ask_event_wants_loop", "plan_asks", "run_one_cycle"]
