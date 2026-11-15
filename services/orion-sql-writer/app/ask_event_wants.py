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

Expected outcome (attention patch 4a, docs/superpowers/specs/2026-11-12-attention-camera-asks-bind-design.md):
each ask is a ``vision.ask`` act graded by Juniper's answer (orion/grammar/camera_ask_acts.py), so the want
carries ``outcome_value`` = the answered share of the same ask RULE (confirm, same, face, office, vehicle,
sighting) over 30 days, shrunk toward the family's answered share ``m``: ``(answered + 2m) / (closed + 2)`` (issue #3350). Once the family has 30 graded closes the market ranks
``salience x outcome``: a vehicle ask Juniper cannot answer (live 2026-11-13: 5 of 47) ranks low however often it was seen.
Juniper's own walkway-story naming cards are not Orion's asks and are in neither the asks nor the history.

SHADOW: nothing obeys the decision. Prime reads the funded ones as its concerns (orion-mind
``PRIME_CONCERNS_FROM_EVENT_WANTS``).

Attention patch 4b (app/ask_bind.py, ``ORION_PHYSIOLOGY_BID_CAMERA_ASKS=on``): the ask bids BEFORE its card opens,
and a card opened on a market answer carries ``want:<want_id>`` in ``evidence_refs``. This loop posts nothing for
such a card (one want per ask). It still posts for a card with no stamp: everything under ``shadow``, and under
``on`` a card the market could not answer at open (down or timed out) or one opened before 4b.

Env: ``ASK_EVENT_WANTS_INTERVAL_SEC`` (default 300; 0 disables), ``ASK_EVENT_WANTS_LOOKBACK_HOURS``
(default 24: an open ask older than this is not posted).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Sequence

from orion.grammar.camera_ask_acts import (
    ACTOR as ASK_ACTOR,
    bid_want_ref,
    ORION_MADE_CAMERA_ASK_SQL,
    OUTCOME_WINDOW,
    ask_rule,
    rule_from_test_before,
    rule_outcome_values,
)
from orion.physiology.event_wants import SOURCE_HISTORY, event_want, event_want_id, source_salience

logger = logging.getLogger("sql-writer.ask_event_wants")

SOURCE = "camera_ask"
FAMILY = "vision.ask"
VERB = "ask"
TIER = 2
PRODUCER = "orion-sql-writer.ask_events"
UNAVAILABLE = "physiology_unavailable"
# Camera asks ORION opened in [:since, :until), each with the moving sightings of its individual up to its
# open and that individual's kind (for the ask's rule). Juniper's own story-naming cards are not Orion's asks:
# they are excluded from the asks AND from the salience history (attention patch 4a).
ASKS_SQL = f"""
WITH a AS (
    SELECT a.ask_id, a.source_kind, a.status, a.created_at, a.expires_at, a.evidence_refs,
           split_part(a.source_ref, '|', 1) AS iid
      FROM orion_ask a
     WHERE {ORION_MADE_CAMERA_ASK_SQL} AND a.created_at >= :since AND a.created_at < :until)
SELECT a.ask_id, a.source_kind, a.status, a.created_at, a.expires_at, a.evidence_refs, a.iid,
       (SELECT i.kind FROM vision_individual i WHERE i.individual_id = a.iid) AS individual_kind,
       (SELECT count(*) FROM vision_individual_sighting v
         WHERE v.individual_id = a.iid AND v.started_at <= a.created_at AND NOT v.stationary) AS sightings
  FROM a ORDER BY a.created_at
"""

# Juniper's grades of camera asks: each graded vision.ask act's rule (test_before camera_ask:<rule>) and outcome,
# selected EXACTLY as the market selects the family's graded acts for its mean (orion/physiology/appetite.py
# build_family_appetite: acts OPENED in the last 30 days, up to now, with a producer or resolver outcome). The rule's
# expected answer shrinks toward that same family share (issue #3350), so the two must not drift: a rule with no
# closes reads the family share and must tie the market's mean, not fall below it. A withdrawn ask (``expired``)
# is not a grade.
GRADES_SQL = (
    "SELECT test_before, outcome_value FROM grammar_acts WHERE actor = :actor "
    "AND outcome_source IN ('producer', 'resolver') AND outcome_value IS NOT NULL "
    "AND coalesce(opened_at, closed_at) >= :since AND coalesce(opened_at, closed_at) <= :now"
)

# Attention patch 4b: a card whose bid-at-open the market DID decide, though the client gave up (0.5 s) and opened it
# unstamped. The binding producer's row names the card (event_ref orion_ask:<ask_id>): that is its one want.
BID_AT_OPEN_SQL = (
    "SELECT DISTINCT decision->'want'->>'event_ref' FROM want_ledger WHERE producer = :producer "
    "AND reason <> 'physiology_unavailable' AND decided_at >= :since "
    "AND decision->'want'->>'event_ref' = ANY(:refs)"
)
BINDING_PRODUCER = "orion-sql-writer.ask_events.binding"

POSTED_SQL = "SELECT DISTINCT want_id FROM want_ledger WHERE reason <> 'physiology_unavailable' AND want_id = ANY(:ids) AND decided_at >= :since"


def _utc(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


def plan_asks(rows: Sequence[Any], *, now: datetime, lookback: timedelta,
              outcomes: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
    """The OPEN camera asks opened inside ``lookback``, each with its salience (None: too little
    history), its rule, and the rule's expected outcome (``outcomes``: rule -> answered share, from
    ``rule_outcome_values``; None: no outcome on the want). Each ask is ranked against the camera asks of
    the 30 days before it. Pure."""
    asks = sorted(({"ask_id": str(r["ask_id"]), "kind": str(r["source_kind"]), "status": str(r["status"]),
                    "at": _utc(r["created_at"]), "expires_at": _utc(r["expires_at"]), "iid": str(r["iid"] or ""),
                    "sightings": int(r["sightings"] or 0), "bid_want": bid_want_ref(r.get("evidence_refs")),
                    "rule": ask_rule(str(r["source_kind"]), r.get("evidence_refs"), r.get("individual_kind"))}
                   for r in rows if r["created_at"] is not None),
                  key=lambda a: a["at"])
    out = []
    for a in asks:
        if a["status"] != "open" or a["at"] < now - lookback or a["at"] > now:
            continue
        if a["expires_at"] is not None and a["expires_at"] <= now:
            continue
        history = [h["sightings"] for h in asks if a["at"] - SOURCE_HISTORY <= h["at"] < a["at"]]
        out.append({**a, "salience": source_salience(a["sightings"], history), "history_n": len(history),
                    "outcome_value": None if outcomes is None else outcomes.get(a["rule"])})
    return out


def run_one_cycle(*, postgres_uri: str, lookback: timedelta, posted: set[str], bid: Callable[[Any], Any],
                  now: Optional[datetime] = None, engine: Any = None, graded: bool = True) -> dict[str, Any]:
    """Read, plan and bid once (sync: runs in a worker thread). ``posted`` is updated in place.
    ``graded`` (SQL_WRITER_ACT_CAMERA_ASKS_ENABLED): each want carries its rule's expected outcome."""
    from sqlalchemy import create_engine, text

    ts = now or datetime.now(timezone.utc)
    eng = engine or create_engine(postgres_uri, pool_pre_ping=True)
    summary: dict[str, Any] = {"posted": 0, "no_history": 0, "already": 0, "unanswered": 0, "bid_at_open": 0}
    try:
        with eng.connect() as conn:
            rows = conn.execute(text(ASKS_SQL), {"since": ts - lookback - SOURCE_HISTORY, "until": ts}).mappings().all()
            outcomes = None
            if graded:
                grades = conn.execute(text(GRADES_SQL), {"actor": ASK_ACTOR, "since": ts - OUTCOME_WINDOW,
                                                         "now": ts}).fetchall()
                outcomes = rule_outcome_values((rule_from_test_before(g[0]) or "", float(g[1])) for g in grades)
            plan = plan_asks(rows, now=ts, lookback=lookback, outcomes=outcomes)
            ids = [event_want_id(SOURCE, p["ask_id"]) for p in plan]
            fresh = [i for i in ids if i not in posted]
            if fresh:
                posted |= {str(r[0]) for r in conn.execute(text(POSTED_SQL), {"ids": fresh,
                                                                              "since": ts - lookback - timedelta(days=1)})}
            unstamped = [f"orion_ask:{p['ask_id']}" for p, i in zip(plan, ids)
                         if p.get("bid_want") is None and i not in posted]
            if unstamped:
                late = {str(r[0]) for r in conn.execute(text(BID_AT_OPEN_SQL), {
                    "producer": BINDING_PRODUCER, "refs": unstamped, "since": ts - lookback - timedelta(days=1)})}
                for p in plan:
                    if f"orion_ask:{p['ask_id']}" in late:
                        p["bid_want"] = "ledger"
    finally:
        if engine is None:
            eng.dispose()
    for p, want_id in zip(plan, ids):
        if p.get("bid_want") is not None:
            # Attention patch 4b: this card bid at open (app/ask_bind.py): stamped want:<id>, or the market's row for
            # it is in want_ledger though the client timed out. Its want is that bid; one want per ask, so none here.
            summary["bid_at_open"] += 1
            continue
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
                          latest=p["expires_at"], object_ref=f"vision_individual:{p['iid']}" if p["iid"] else None,
                          outcome_value=p["outcome_value"])
        decision = bid(want)
        if decision is None or decision.reason == UNAVAILABLE:
            summary["unanswered"] += 1       # no answer, or the market could not price it: retried next cycle
            continue
        posted.add(want_id)
        summary["posted"] += 1
        logger.info("ask_event_want ask=%s kind=%s rule=%s sightings=%d salience=%.3f expected_answer=%s outcome=%s "
                    "reason=%s value_rank=%s want_id=%s", p["ask_id"], p["kind"], p["rule"], p["sightings"],
                    p["salience"], p["outcome_value"], decision.outcome, decision.reason, decision.value_rank, want_id)
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
    graded = bool(getattr(settings, "sql_writer_act_camera_asks_enabled", True))
    engine = create_engine(postgres_uri, pool_pre_ping=True, pool_size=1, max_overflow=0)
    posted: set[str] = set()
    logger.info("ask_event_wants_loop starting interval_sec=%.0f lookback_h=%.1f", interval,
                lookback.total_seconds() / 3600)
    while True:
        try:
            summary = await asyncio.to_thread(run_one_cycle, postgres_uri=postgres_uri, lookback=lookback,
                                              posted=posted, bid=bid, engine=engine, graded=graded)
            if summary["posted"] or summary["no_history"] or summary["unanswered"]:
                logger.info("[ASK_EVENT_WANTS] tick %s", summary)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one bad cycle must not stop the loop
            logger.warning("ask_event_wants_cycle_failed error=%s", exc)
        await asyncio.sleep(interval)


__all__ = ["ask_event_wants_loop", "plan_asks", "run_one_cycle"]
