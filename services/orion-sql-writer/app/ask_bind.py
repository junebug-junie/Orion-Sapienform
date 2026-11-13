"""Camera asks bid BEFORE the card opens, and obey the market (attention patch 4b).

Design: docs/superpowers/specs/2026-11-12-attention-camera-asks-bind-design.md, section 4b, with the decisions on
issues #3285 and #3286:

* **One bid site.** Every camera opener (app/vision_individuals.py: same-person, face, slow sighting rule,
  "Is this <name>?"; app/vision_handoff_loop.py door/walkway "same person?"; app/vision_office_sessions.py) calls
  ``open_camera_ask`` instead of inserting the card itself. It calls ``ask_bind`` and inserts only when the bid
  admits. A static test (tests/test_ask_bind.py) refuses any other ``_insert_ask`` caller.
* **The want.** Family ``vision.ask``, producer ``orion-sql-writer.ask_events.binding`` (in
  ``orion.physiology.binding.BINDING_PRODUCERS``, so the market treats it as obeyed). Deterministic want id per
  CANDIDATE (``candidate_key``: rule, kind, subject), so a held candidate re-bids as the same want. Salience and
  ``outcome_value`` exactly as patch 4a's producer (app/ask_event_wants.py): the individual's moving sightings as a
  strict-below percentile of 30 days of Orion-made asks, and the rule's answered share. ``event_ref`` is
  ``orion_ask:<ask_id>`` of the card it would open: Prime's concern menu (orion-durable-runs
  app/event_wants_menu.py) labels and expires event wants by that ref, and the funded bid is the candidate's last.
* **Obey.** ``orion.physiology.binding.bind_verdict``. Funded: open the card, stamped ``want:<want_id>`` so the
  5-minute shadow loop posts no second want for it. No answer (market down, 0.5 s timeout): open the card, as
  every binding producer does; no stamp, so the shadow loop still records one want for it. Held (sated, a Prime
  charge): do not open. The candidate comes back on the opener's next tick by itself; ``_held`` keeps its next
  bid time (``rebid_after_sec``: 600 s doubling to 3 h), so it is not re-bid every 60 s.
* **No deadline** (issue #3285). A held ask keeps bidding until it is funded or stops being a candidate (its
  opener no longer proposes it: it then just drops, it was never shown). ``bind_verdict`` admits at ``end``, so it
  gets a rolling end one day past now that never arrives.
* **Ranking** is the market's: family hunger x expected answer (issue #3286). No vision.ask exception.
* **The daily caps stay** a hard ceiling: the openers check them before they ask; a held candidate takes no slot.
* **Juniper's own cards never bid.** Her walkway-story names are inserted by Hub, already answered, and never
  pass through here; ``ask_bind`` refuses one anyway (``is_juniper_card``).

Flag ``ORION_PHYSIOLOGY_BID_CAMERA_ASKS``: ``on`` (default) binds; ``shadow`` is the behaviour before 4b (open
first, the shadow loop posts the want after). Read at call time.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from orion.grammar.camera_ask_acts import (
    ACTOR as ASK_ACTOR,
    BID_WANT_REF_PREFIX,
    CANDIDATE_SOURCE,
    OUTCOME_WINDOW,
    ask_rule,
    candidate_key,
    is_juniper_card,
    is_orion_camera_ask,
    rebid_after_sec,
    rule_from_test_before,
    rule_outcome_values,
)
from orion.physiology.binding import NO_ANSWER_REASONS, BindVerdict, bind_verdict
from orion.physiology.event_wants import SOURCE_HISTORY, event_want, event_want_id, source_salience

logger = logging.getLogger("sql-writer.ask_bind")

FAMILY = "vision.ask"
VERB = "ask"
TIER = 2
PRODUCER = "orion-sql-writer.ask_events.binding"   # == app.ask_event_wants.BINDING_PRODUCER (tested)
MODE_ON = "on"
MODE_SHADOW = "shadow"
# A rolling window end that never arrives: a held ask has no deadline (issue #3285).
NO_DEADLINE = timedelta(days=1)
# History (30 days of asks' sightings) and per-rule outcomes are re-read at most this often.
CACHE_TTL_SEC = 300.0
# A held entry whose retry_at is this long past is forgotten (its candidate dropped). Measured from retry_at, not
# from the last proposal: a candidate back within the window resumes its backoff where it was (intended).
HELD_FORGET = timedelta(days=2)
# At most this many FRESH market bids per rolling BID_BUDGET_WINDOW, per process (review finding: the bids run inside
# the opener's DB transaction, and a held candidate no longer stops an opener's loop, so after a restart, with
# ``_held`` empty, one tick could bid ~130 candidates at up to 0.5 s each). Over budget, a candidate waits one
# window with no bid and no backoff step (``bid_budget``); it is bid on a later tick, in the opener's own order.
BID_BUDGET = 12
BID_BUDGET_WINDOW = timedelta(seconds=60)

SIGHTINGS_SQL = ("SELECT count(*) FROM vision_individual_sighting WHERE individual_id = :i AND started_at <= :t "
                 "AND NOT stationary")
KIND_SQL = "SELECT kind FROM vision_individual WHERE individual_id = :i"


@dataclass
class _Held:
    retry_at: datetime
    n_held: int
    first_seen: datetime


_lock = threading.Lock()
_held: dict[str, _Held] = {}
_cache: dict[str, Any] = {"at": None, "history": None, "outcomes": None}
_bids: list[datetime] = []          # times of this process's fresh bids inside BID_BUDGET_WINDOW


def reset_state() -> None:
    """Forget held candidates and the cache (tests; a process restart does the same)."""
    with _lock:
        _held.clear()
        _bids.clear()
        _cache.update(at=None, history=None, outcomes=None)


def held_snapshot() -> dict[str, _Held]:
    with _lock:
        return dict(_held)


def mode() -> str:
    from app.settings import settings

    v = str(getattr(settings, "orion_physiology_bid_camera_asks", MODE_ON) or MODE_ON).strip().lower()
    return MODE_SHADOW if v == MODE_SHADOW else MODE_ON


def _graded() -> bool:
    from app.settings import settings

    return bool(getattr(settings, "sql_writer_act_camera_asks_enabled", True))


def _first_individual(source_ref: str) -> str:
    return str(source_ref or "").split("|", 1)[0]


def _history_and_outcomes(conn, now: datetime, graded: bool) -> tuple[list[int], Optional[dict[str, float]]]:
    """The sightings of every Orion-made camera ask of the 30 days before ``now`` and each rule's expected
    answer (None when ungraded), cached CACHE_TTL_SEC. Same SQL as patch 4a's producer."""
    from sqlalchemy import text

    from app.ask_event_wants import ASKS_SQL, GRADES_SQL

    # Cached on the monotonic clock, not on ``now``: a test or replay with an old ``now`` reads today's cache. Two
    # threads may both refresh at once; harmless (same reads, last write wins).
    mono = time.monotonic()
    with _lock:
        if _cache["at"] is not None and mono - _cache["at"] < CACHE_TTL_SEC and _cache["history"] is not None:
            return _cache["history"], (_cache["outcomes"] if graded else None)
    rows = conn.execute(text(ASKS_SQL), {"since": now - SOURCE_HISTORY, "until": now}).mappings().all()
    history = [int(r["sightings"] or 0) for r in rows]
    grades = conn.execute(text(GRADES_SQL), {"actor": ASK_ACTOR, "since": now - OUTCOME_WINDOW}).fetchall()
    outcomes = rule_outcome_values((rule_from_test_before(g[0]) or "", float(g[1])) for g in grades)
    with _lock:
        _cache.update(at=mono, history=history, outcomes=outcomes)
    return history, (outcomes if graded else None)


def in_backoff(rule: str, source_kind: str, source_ref: str, now: datetime) -> bool:
    """True while this candidate is held and its next bid is not due. Openers call it BEFORE their picture work,
    so a held candidate costs no sighting reads on every 60 s tick (review finding)."""
    want_id = event_want_id(CANDIDATE_SOURCE, candidate_key(rule, source_kind, source_ref))
    with _lock:
        held = _held.get(want_id)
    return held is not None and now < held.retry_at


def ask_bind(conn, ask: Any, *, now: datetime, bid: Optional[Callable[[Any], Any]] = None,
             individual_kind: Optional[str] = None) -> BindVerdict:
    """Bid one candidate ask; admit (open the card) or hold. ``ask``: the ``OrionAskV1`` the opener built.
    ``bid``: the market call (default ``orion.physiology.client.bid``). Sync: openers run in worker threads."""
    from sqlalchemy import text

    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    if is_juniper_card(ask.evidence_refs, None):
        return BindVerdict(True, "juniper_card")           # her own card: never a want (attention patch 4a)
    if not is_orion_camera_ask(ask.source_kind, ask.evidence_refs, None):
        return BindVerdict(True, "not_a_camera_ask")
    if mode() != MODE_ON:
        return BindVerdict(True, "shadow")                 # the shadow loop posts the want after the card opens
    iid = _first_individual(ask.source_ref)
    kind = individual_kind
    if kind is None and iid:
        kind = conn.execute(text(KIND_SQL), {"i": iid}).scalar()
    rule = ask_rule(str(ask.source_kind), ask.evidence_refs, kind)
    # The face same-person opener and the handoff opener share the ``same`` key for a pair: one backoff for both
    # (intended: it is one question). Two openers bidding the same pair at once is a double bid at worst; the
    # open-card unique index stops a double card.
    key = candidate_key(rule, str(ask.source_kind), str(ask.source_ref))
    want_id = event_want_id(CANDIDATE_SOURCE, key)
    with _lock:
        for k in [k for k, h in _held.items() if now - h.retry_at > HELD_FORGET]:
            del _held[k]
        held = _held.get(want_id)
    if held is not None and now < held.retry_at:
        return BindVerdict(False, "backoff", retry_at=held.retry_at, detail={"want_id": want_id, "rule": rule,
                                                                             "n_held": held.n_held})
    with _lock:
        _bids[:] = [t for t in _bids if now - t < BID_BUDGET_WINDOW]
        over = len(_bids) >= BID_BUDGET
        if not over:
            _bids.append(now)
    if over:
        return BindVerdict(False, "bid_budget", retry_at=now + BID_BUDGET_WINDOW,
                           detail={"want_id": want_id, "rule": rule})
    graded = _graded()
    history, outcomes = _history_and_outcomes(conn, now, graded)
    sightings = int(conn.execute(text(SIGHTINGS_SQL), {"i": iid, "t": now}).scalar() or 0) if iid else 0
    salience = source_salience(sightings, history)
    if salience is None:
        # Too little history to rank an event (fewer than 20 asks in 30 days): no want, open as before 4b.
        logger.info("camera_ask_bid ask=%s rule=%s want_id=%s verdict=admit why=no_answer:no_history history_n=%d",
                    ask.ask_id, rule, want_id, len(history))
        return BindVerdict(True, "no_answer:no_history", detail={"want_id": want_id, "rule": rule})
    first_seen = held.first_seen if held is not None else now
    want = event_want(source=CANDIDATE_SOURCE, key=key, family=FAMILY, producer=PRODUCER, verb=VERB, tier=TIER,
                      event_ref=f"orion_ask:{ask.ask_id}", salience=salience, opened_at=first_seen,
                      latest=ask.expires_at, object_ref=f"vision_individual:{iid}" if iid else None,
                      outcome_value=None if outcomes is None else outcomes.get(rule))
    if bid is None:
        from orion.physiology.client import bid as _client_bid

        bid = _client_bid
    try:
        decision = bid(want)
    except Exception as exc:  # noqa: BLE001 -- a broken market call is no answer: admit, as bind_verdict does
        logger.info("camera_ask_bid_failed ask=%s err=%s", ask.ask_id, type(exc).__name__)
        decision = None
    verdict = bind_verdict(decision, now, now + NO_DEADLINE)
    answered = decision is not None and str(getattr(decision, "reason", "") or "") not in NO_ANSWER_REASONS
    detail = {**verdict.detail, "want_id": want_id, "rule": rule, "salience": round(salience, 4),
              "expected_answer": want.outcome_value, "market_answered": answered}
    if verdict.admit:
        with _lock:
            _held.pop(want_id, None)
        logger.info("camera_ask_bid ask=%s rule=%s want_id=%s verdict=admit why=%s salience=%.3f expected_answer=%s "
                    "held_before=%d", ask.ask_id, rule, want_id, verdict.why, salience, want.outcome_value,
                    held.n_held if held is not None else 0)
        return BindVerdict(True, verdict.why, detail=detail)
    n = (held.n_held if held is not None else 0) + 1
    market_wait = (verdict.retry_at - now).total_seconds() if verdict.retry_at is not None else None
    retry_at = now + timedelta(seconds=rebid_after_sec(n, market_wait))
    with _lock:
        _held[want_id] = _Held(retry_at=retry_at, n_held=n, first_seen=first_seen)
    # history_n on the held path too: held asks open no card and so stop feeding the salience history; if it ever
    # falls under 20, asks open unbid (no_history). Live: ~450 Orion asks in 30 days, far from it (review nit).
    logger.info("camera_ask_bid ask=%s rule=%s want_id=%s verdict=held why=%s salience=%.3f expected_answer=%s "
                "n_held=%d retry_at=%s history_n=%d", ask.ask_id, rule, want_id, verdict.why, salience,
                want.outcome_value, n, retry_at.isoformat(), len(history))
    return BindVerdict(False, verdict.why, retry_at=retry_at, detail={**detail, "n_held": n})


def open_camera_ask(conn, ask: Any, *, now: datetime, bid: Optional[Callable[[Any], Any]] = None,
                    individual_kind: Optional[str] = None) -> Optional[Any]:
    """The one way a camera opener opens a card: bid, then insert only when admitted. Returns the inserted
    ``OrionAskV1`` (with its ``want:<id>`` stamp when the market answered), or None (held, or the insert lost
    to an open card for the same subject)."""
    from app.vision_individuals import _insert_ask

    verdict = ask_bind(conn, ask, now=now, bid=bid, individual_kind=individual_kind)
    if not verdict.admit:
        return None
    if verdict.detail.get("market_answered") and verdict.detail.get("want_id"):
        ask = ask.model_copy(update={"evidence_refs": list(ask.evidence_refs)
                                     + [f"{BID_WANT_REF_PREFIX}{verdict.detail['want_id']}"]})
    return ask if _insert_ask(conn, ask) else None


__all__ = ["BID_BUDGET", "FAMILY", "in_backoff", "MODE_ON", "MODE_SHADOW", "PRODUCER", "ask_bind", "held_snapshot", "mode", "open_camera_ask",
           "reset_state"]
