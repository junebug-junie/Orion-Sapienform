#!/usr/bin/env python3
"""Eval: the two event sources and Prime's intake, replayed over real history. READ-ONLY.

Attention as the want competition, patch 2 (docs/superpowers/specs/2026-11-04-attention-as-want-
competition-design.md). No event want existed before this patch, so the replay rebuilds them from the
source rows with the SAME pure rules the producers and the market use:

1. **Salience per source** (``orion.physiology.event_wants.saliences_over_own_history``):
   * reading: each finished read (``world_pulse_read_seed``, ``stage2_status = 'done'``), magnitude = the
     links it made (accepted relationship claims), as services/orion-hub/scripts/reading_event_wants.py;
   * camera asks: each camera ask in ``orion_ask``, magnitude = moving sightings of its individual when it
     opened, as services/orion-sql-writer/app/ask_event_wants.py. Only asks still open when the producer
     would have polled (``POLL_DELAY``) are posted; the distribution is reported for both.
   Acceptance check 1: most events have low salience (share under 0.5 is over half, per source).
2. **The market's decision** on each posted want, from the act history at that moment: hunger
   (``appetite.hunger``), intensity = max(hunger, salience), and the event rank (``appetite.rank_value``
   with ``event=True``) against the source's earlier event wants. Camera asks have no lanes; reading
   declares lanes since patch 3a, but its look states that it holds no GPU slot (patch 3b:
   ``reading_event_wants.LOOK_COST``). Neither has a fixed cost, so no charge applies and Choice funds
   every want whose rank is above 0 (asserted, not assumed).
   Reported beside the patch-1 rank (against the hunger-only act history) to show the saturation fix.
3. **Prime's intake** (orion-mind ``PRIME_CONCERNS_FROM_EVENT_WANTS``): at each real Prime thought and at
   each real concern bet of the window, how many funded, unexpired event wants would the frame show
   (decided within ``PRIME_EVENT_WANTS_HOURS``, before their window end; an ask answered by then is gone).
   Gate (the flag ships ON only if it passes): at least ``COVER_MIN`` of the moments Prime really added a
   concern (its concern bets) would have shown at least one funded event. The share of all thoughts is
   reported as a diagnostic: a thought with no funded event keeps its held concerns, it cannot add one.

Approximations, stated: the market is replayed with today's families.yaml and the act history as it is
now stored; weights do not enter (they scale net, not funded/deferred, when the charge is 0); a want is
bid ``POLL_DELAY`` after its event (the producers poll every 300 s); Prime's own wake and refractory
rules are unchanged by the flag, so its real thought times are the right sample.

Two lanes:

* ``--fixture`` (default; deterministic, no database): salience rest point, event rank not saturated,
  and the starvation count on a constructed day.
* ``--live --days 14``: the real tables.

**Camera-asks gate** (``--camera-asks``, attention patch 4a; acceptance check 3 of
docs/superpowers/specs/2026-11-12-attention-camera-asks-bind-design.md). Patch 4b (an ask bids BEFORE its
card opens, and a held ask is not opened) may ship only if this replay passes. Over the real Orion-made
camera asks (Juniper's own walkway-story naming cards excluded), each ask is bid at its real open time with
4a's value: salience (moving sightings, as the producer) and ``outcome_value`` = its rule's answered share
over the 30 days before the bid (``orion.grammar.camera_ask_acts.rule_outcome_values``), against acts
rebuilt from the asks themselves (open at the ask, closed by ``ask_grade``) as the act history stood at the
bid. 4b's rule, as decided on issue #3285 (no deadline): funded opens the card; held re-bids with the live
backoff (``rebid_after_sec``: 600 s doubling to 3 h) until funded. Held means "re-bid later, may open later",
not "dropped at a window end". An ask is followed ``LOSS_HORIZON`` (7 d) after its first bid, or to the replay's
end. Gate: at least 70% of vehicle asks held (at their first bid, and still unopened through the horizon); at most 10% of confirm and same-person asks
held at all; answered asks per day fall by at most 10% (an answered ask not opened within the horizon is a lost
answer). Approximations, stated: an ask held and opened later is assumed answered as it really was; Juniper's
real answers decided the history every bid sees (4b would have changed some); a held candidate is assumed to
stay a candidate for the whole horizon; the daily caps are not replayed (a later open could meet a full cap);
no Prime charge (camera asks carry none today).

Since issue #3364 the live lane ranks every bid against the market's own reference, the family's past event wants in
``want_ledger`` as the market reads them (with this replay's first bids standing in only before the ledger
existed), through the market's 300 s snapshot. It re-decides every live 4b bid and must match each one under the
rule the market ran then (``replay_reproduces_live_bids``), and it reports the next bid of each rule against today's
ledger (``next_bid_by_rule``). ``--export-parity-fixture PATH`` writes those live inputs, anonymised, for the
regression test (tests/test_camera_asks_replay_parity.py).

    PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/event_wants_replay.py [--live --days 14]
    PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/event_wants_replay.py --camera-asks [--live --days 30]
    # read-only, against the live database from the host:
    ORION_ANALYSIS_PSQL="psql postgresql://postgres:postgres@localhost:55432/conjourney -At -q -v ON_ERROR_STOP=1" \
        PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/event_wants_replay.py --camera-asks --live --days 30
"""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC), str(ROOT / "scripts" / "analysis")]

from orion.physiology.appetite import ActRecord, build_family_appetite, hunger, rank_value, want_intensity  # noqa: E402
from orion.physiology.event_wants import SOURCE_HISTORY, saliences_over_own_history  # noqa: E402
from orion.physiology.families import FamiliesConfig, load_families, parse_families  # noqa: E402

READ_FAMILY = "hub.world_pulse_read"
ASK_FAMILY = "vision.ask"
POLL_DELAY = timedelta(seconds=150)
PRIME_HOURS = 24.0
COVER_MIN = 0.95
LOW = 0.5


def _ts(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def _q(vals: Sequence[float], q: float) -> Optional[float]:
    v = sorted(vals)
    if not v:
        return None
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return round(v[lo] * (1 - (pos - lo)) + v[hi] * (pos - lo), 4)


def distribution(vals: Sequence[float]) -> dict:
    n = len(vals)
    if not n:
        return {"n": 0}
    return {"n": n, "share_zero": round(sum(1 for v in vals if v == 0) / n, 4),
            "share_under_0_5": round(sum(1 for v in vals if v < LOW) / n, 4),
            "share_at_least_0_9": round(sum(1 for v in vals if v >= 0.9) / n, 4),
            "p25": _q(vals, 0.25), "p50": _q(vals, 0.5), "p75": _q(vals, 0.75), "p90": _q(vals, 0.9),
            "max": round(max(vals), 4)}


@dataclasses.dataclass
class Event:
    family: str
    ref: str
    at: datetime                       # the event (read finished / ask opened)
    magnitude: float
    ends_at: Optional[datetime] = None  # an ask answered/expired; None: the family window decides
    posted: bool = True                # open when the producer polled (asks); reads always
    salience: Optional[float] = None
    intensity: Optional[float] = None
    rank_new: Optional[float] = None
    rank_old: Optional[float] = None
    funded: bool = False
    expires: Optional[datetime] = None


# Families whose event wants state an empty gpu_slot_s_by_lane (services/orion-hub/scripts/
# reading_event_wants.py LOOK_COST, patch 3b): no lane placement, no lane charge, no capacity check.
NO_SLOT_FAMILIES = frozenset({READ_FAMILY})


def simulate(families: FamiliesConfig, events: list[Event], acts: Sequence[ActRecord]) -> list[Event]:
    """Salience, then the market's event-want decision, in time order. Mutates and returns ``events``."""
    for fam in {e.family for e in events}:
        cfg = families.get(fam)
        assert cfg is not None, fam
        # No lane charge (no lanes, or the producer states the act holds no slot) and no fixed cost:
        # Choice charges nothing, so it funds iff the rank is above 0.
        assert (not cfg.lanes or fam in NO_SLOT_FAMILIES) and not cfg.fixed_cost, \
            f"{fam} has a charge; this replay assumes none"
        mine = sorted((e for e in events if e.family == fam), key=lambda e: e.at)
        for e, s in zip(mine, saliences_over_own_history([(e.at, e.magnitude) for e in mine])):
            e.salience = s
        fam_acts = [a for a in acts if a.family == fam]
        closes = sorted(a.closed_at for a in fam_acts if a.closed_at is not None)
        day_app: dict[Any, Any] = {}
        history: list[tuple[datetime, float]] = []   # (bid_at, intensity) of earlier posted event wants
        for e in mine:
            if not e.posted or e.salience is None:
                continue
            bid_at = e.at + POLL_DELAY
            day = bid_at.date()
            if day not in day_app:
                day_app[day] = build_family_appetite(cfg, fam_acts, datetime.combine(day, datetime.min.time(),
                                                                                    tzinfo=timezone.utc))
            i = bisect.bisect_left(closes, bid_at)
            app = dataclasses.replace(day_app[day], last_close_at=closes[i - 1] if i else None)
            h = hunger(app, bid_at, opened_at=e.at)
            inten, _ = want_intensity(h.intensity, e.salience)
            ev_hist = [v for t, v in history if bid_at - SOURCE_HISTORY <= t < bid_at]
            e.rank_new, _, _ = rank_value(dataclasses.replace(app, event_values=tuple(ev_hist)), inten, None, event=True)
            e.rank_old, _, _ = rank_value(app, inten, None)
            e.intensity = inten
            e.funded = e.rank_new > 0
            window_end = e.at + timedelta(seconds=cfg.deferrable_window_s)
            e.expires = min(x for x in (window_end, e.ends_at) if x is not None)
            history.append((bid_at, inten))
    return events


def prime_cover(events: Sequence[Event], moments: Sequence[datetime], *, hours: float = PRIME_HOURS) -> dict:
    """For each moment: funded event wants decided within ``hours`` and not yet expired."""
    funded = [e for e in events if e.funded]
    counts = []
    salient = []
    for t in moments:
        live = [e for e in funded if e.at + POLL_DELAY <= t < e.expires and t - (e.at + POLL_DELAY) <= timedelta(hours=hours)]
        counts.append(len(live))
        salient.append(sum(1 for e in live if (e.salience or 0) >= LOW))
    n = len(moments)
    return {"moments": n, "share_with_an_event": round(sum(1 for c in counts if c) / n, 4) if n else None,
            "share_with_a_salient_event": round(sum(1 for c in salient if c) / n, 4) if n else None,
            "events_shown_p50": _q(counts, 0.5), "events_shown_max": max(counts) if counts else None}


def report(families: FamiliesConfig, events: list[Event], acts: Sequence[ActRecord], *, start: datetime,
           thoughts: Sequence[datetime], concern_bets: Sequence[datetime]) -> dict:
    simulate(families, events, acts)
    out: dict[str, Any] = {"sources": {}}
    for fam in sorted({e.family for e in events}):
        win = [e for e in events if e.family == fam and e.at >= start and e.salience is not None]
        posted = [e for e in win if e.posted]
        ranked = [e for e in posted if e.rank_new is not None]
        hi = [e for e in ranked if e.salience >= LOW]
        lo = [e for e in ranked if e.salience < LOW]
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        sat = [e for e in ranked if e.salience >= 0.8]
        ordinary = [e for e in posted if e.salience == 0]
        out["sources"][fam] = {
            "events": len(win), "posted": len(posted),
            "salience_all_events": distribution([e.salience for e in win]),
            "salience_posted": distribution([e.salience for e in posted]),
            "funded_share": round(sum(e.funded for e in posted) / len(posted), 4) if posted else None,
            # Event rank (patch-1 review): ranked against the hunger-only act history (old) vs the source's
            # own earlier event wants (new). Separation: mean rank of salient minus non-salient events.
            "rank_old": distribution([e.rank_old for e in ranked]),
            "rank_new": distribution([e.rank_new for e in ranked]),
            "separation_old": (round(mean([e.rank_old for e in hi]) - mean([e.rank_old for e in lo]), 4)
                               if hi and lo else None),
            "separation_new": (round(mean([e.rank_new for e in hi]) - mean([e.rank_new for e in lo]), 4)
                               if hi and lo else None),
            "very_salient_events": len(sat),
            "very_salient_rank_at_least_0_95_old": round(sum(1 for e in sat if e.rank_old >= 0.95) / len(sat), 4) if sat else None,
            "very_salient_rank_at_least_0_95_new": round(sum(1 for e in sat if e.rank_new >= 0.95) / len(sat), 4) if sat else None,
            # Does "funded" filter anything? The share of salience-0 (ordinary) event wants the market funds.
            "ordinary_events": len(ordinary),
            "ordinary_funded_share": round(sum(e.funded for e in ordinary) / len(ordinary), 4) if ordinary else None,
        }
    out["prime_thoughts"] = prime_cover(events, thoughts)
    out["prime_concern_bets"] = prime_cover(events, concern_bets)
    src = out["sources"]
    checks = {f"{fam}:most_events_low": v["salience_all_events"].get("share_under_0_5", 0) > 0.5 for fam, v in src.items()}
    # Separation (old vs new) is a diagnostic only: a saturating rank separates perfectly, so "separates
    # better" would penalise the fix. The gate is that very salient events are not pinned near 1.0.
    checks.update({f"{fam}:very_salient_rank_not_saturated": (v["very_salient_rank_at_least_0_95_new"] or 0) <= 0.5
                   for fam, v in src.items()})
    # The flag's gate: at the moments Prime really added a concern, the event list would not have been empty.
    checks["concern_moments_covered"] = ((out["prime_concern_bets"]["share_with_an_event"] or 0) >= COVER_MIN
                                         if concern_bets else True)
    out["checks"] = checks
    # Diagnostic, not a gate: a thought with no funded event keeps its held concerns; it cannot add one.
    out["diagnostics"] = {"thoughts_with_an_event": out["prime_thoughts"]["share_with_an_event"]}
    return out


# --- fixture lane --------------------------------------------------------------------------------

def _fixture() -> int:
    fams = parse_families({"version": 1, "families": {
        READ_FAMILY: {"tier": 2, "deferrable_window_s": 10800, "lanes": []}}})
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    events, acts = [], []
    # 60 days of reads every 2 h; 1 in 6 makes a link, 1 in 30 makes three. Each read closes an act.
    for k in range(60 * 12):
        at = t0 + timedelta(hours=2 * k)
        links = 3 if k % 30 == 0 else (1 if k % 6 == 0 else 0)
        events.append(Event(family=READ_FAMILY, ref=f"r{k}", at=at, magnitude=links))
        acts.append(ActRecord(act_id=f"a{k}", family=READ_FAMILY, opened_at=at - timedelta(minutes=20), closed_at=at))
    start = t0 + timedelta(days=45)
    thoughts = [start + timedelta(minutes=37 * j) for j in range(400)]
    salient_at = next(e.at for e in events if e.at >= start and e.magnitude > 0)
    ordinary_at = next(e.at for e in events if e.at >= start and e.magnitude == 0
                       and not any(0 <= (e.at - x.at).total_seconds() < 4 * 3600 for x in events if x.magnitude > 0))
    r = report(fams, events, acts, start=start, thoughts=thoughts, concern_bets=thoughts[::10])
    s = r["sources"][READ_FAMILY]
    checks = {
        "rest_point_is_zero": s["salience_all_events"]["share_zero"] > 0.75,
        "most_events_low": r["checks"][f"{READ_FAMILY}:most_events_low"],
        "very_salient_rank_not_saturated": s["very_salient_rank_at_least_0_95_new"] < s["very_salient_rank_at_least_0_95_old"],
        "ordinary_events_not_all_funded": (s["ordinary_funded_share"] or 0) < 1.0,
        # Prime sees exactly the funded, unexpired events: 1 h after a salient read there is one; 1 h after
        # an ordinary read (tied with the usual ones, so sated) and long after any read there is none.
        "prime_sees_a_funded_salient_read": prime_cover(events, [salient_at + timedelta(hours=1)])["share_with_an_event"] == 1.0,
        "prime_sees_no_ordinary_read": prime_cover(events, [ordinary_at + timedelta(hours=1)])["share_with_an_event"] == 0.0,
        "a_gap_starves": prime_cover(events, [t0 + timedelta(days=200)])["share_with_an_event"] == 0.0,
    }
    # The fixture's own checks AND every check report() made (none may be hidden).
    # Every check report() made, except the coverage gate: the fixture's synthetic thoughts are not Prime's
    # rhythm (one salient read in six, by construction); its coverage logic is tested by hand above.
    checks.update({f"report:{k}": v for k, v in r["checks"].items() if k != "concern_moments_covered"})
    print(json.dumps({"lane": "fixture", **r, "checks": checks}, indent=2, default=str))
    return 0 if all(checks.values()) else 1


# --- camera-asks gate (patch 4a -> 4b) -------------------------------------------------------------

# A held candidate is followed this long after its first bid: the card's own answerable life (VISION_ASK_EXPIRY_DAYS).
# An answered ask not opened by then counts as a lost answer (conservative: a real candidate may stay one longer).
LOSS_HORIZON = timedelta(days=7)
GATE_VEHICLE_HELD_MIN = 0.70
GATE_ANSWERABLE_HELD_MAX = 0.10
GATE_ANSWERED_LOSS_MAX = 0.10
# The market caches its act and event history for 300 s (orion-durable-runs Market history_ttl_sec), and the opener
# caches its rule outcomes for 300 s (app/ask_bind.py CACHE_TTL_SEC): a bid sees what was known this long before it.
MARKET_SNAPSHOT_LAG = timedelta(seconds=300)


def ask_acts_from_asks(asks: Sequence[Mapping[str, Any]], *, now: datetime) -> list[tuple[ActRecord, str]]:
    """(act, rule) per Orion-made ask, as 4a's ledger would hold it: opened at the ask, closed by
    ``ask_grade`` (still open: no close)."""
    from orion.grammar.camera_ask_acts import ask_grade

    out = []
    for a in asks:
        g = ask_grade(a, now=now)
        out.append((ActRecord(act_id=str(a["ask_id"]), family=ASK_FAMILY, opened_at=a["created_at"],
                              closed_at=g.observed_at if g else None, outcome_value=g.value if g else None,
                              outcome_source=g.source if g else None), str(a["rule"])))
    return out


def _as_of(acts: Sequence[tuple[ActRecord, str]], t: datetime) -> list[tuple[ActRecord, str]]:
    """The act history as it stood at ``t``: acts opened by then, closes after ``t`` not yet known."""
    out = []
    for act, rule in acts:
        if act.opened_at is None or act.opened_at > t:
            continue
        if act.closed_at is not None and act.closed_at > t:
            act = dataclasses.replace(act, closed_at=None, outcome_value=None, outcome_source=None)
        out.append((act, rule))
    return out


def camera_asks_gate(families: FamiliesConfig, asks: Sequence[Mapping[str, Any]], *, start: datetime,
                     end: datetime, ledger: Sequence[Mapping[str, Any]] = (), same_grading: bool = True,
                     snapshot_lag: timedelta = MARKET_SNAPSHOT_LAG, keep_bids: bool = False,
                     probe_rules: Sequence[str] = (), counterfactual: bool = False) -> dict:
    """Replay 4b's rule with 4a's value over ``asks`` (Orion-made only; each has ask_id, rule, created_at,
    expires_at, status, answered_at, answer_detail, sightings). Pure.

    4b's rule as decided on issue #3285: funded opens the card; held re-bids after
    ``orion.grammar.camera_ask_acts.rebid_after_sec`` (600 s doubling to 3 h, the live backoff) with NO deadline,
    until funded or the candidate stops being one. When a real candidate stops is not in the data, so an ask is
    followed until ``LOSS_HORIZON`` after its first bid (the card's own answerable life) or the replay's end.

    **The event-rank reference is the market's (issue #3364).** ``ledger``: the family's real past event wants as
    the market reads them (``want_ledger``, first decision per want: decided_at, intensity, graded, event_ref). A bid
    ranks against the ledger rows decided before its snapshot. Where the market had no ledger yet (bids before its first
    row), this replay's own first bids stand in for it, and only there (review finding: a stand-in kept for 30 days
    after the ledger started gave the 11-14 vehicle bids 239 reference rows, not the market's 120).
    ``counterfactual`` (the gate): as if 4b had bid every ask from the window start, so every ask with no 4b bid in
    the ledger (``BINDING_PRODUCER``) gets a stand-in, counted for its 30 days; the real rows stay (under
    ``same_grading`` the ungraded shadow rows drop out by themselves). With ``same_grading`` (the market's rule since #3364) only the rows
    decided with the family's grading at the bid count; without it every row counts (the rule live from 4b's deploy
    to #3364). Before #3364 the replay ranked against its own first bids ONLY: every one was graded and saturated
    (intensity 1.0), while the live market ranked against 120 ungraded shadow-era rows, three of them near 0. That
    is why the replay held 95.5% of vehicle asks and the market held none.

    **The market's snapshot** (``snapshot_lag``): the market reads acts and the event history through a cache up to
    ``MARKET_SNAPSHOT_LAG`` (300 s) old, and the opener's rule outcomes come from a 300 s cache too. A bid at ``t``
    sees closes, grades and ledger rows up to ``t - snapshot_lag`` (the upper bound of the staleness; it reproduces all
    16 live bids); the appetite is still built at ``t``, as the market does."""
    from orion.grammar.camera_ask_acts import OUTCOME_WINDOW, rebid_after_sec, rule_outcome_values
    from orion.physiology.appetite import same_grading_events
    from orion.physiology.binding import REBID_DEFAULT_SEC

    cfg = families.get(ASK_FAMILY)
    assert cfg is not None and not cfg.lanes and not cfg.fixed_cost, "this replay assumes vision.ask has no charge"
    asks = sorted(asks, key=lambda a: a["created_at"])
    sal = saliences_over_own_history([(a["created_at"], float(a["sightings"] or 0)) for a in asks])
    acts = ask_acts_from_asks(asks, now=end)
    led = sorted(({"at": _ts(r["decided_at"]), "intensity": float(r["intensity"]),
                   "graded": r.get("graded") if isinstance(r.get("graded"), bool) else None,
                   "event_ref": r.get("event_ref"), "producer": r.get("producer")} for r in ledger if r.get("intensity") is not None),
                 key=lambda r: r["at"])
    led_refs = {r["event_ref"] for r in led if r["event_ref"]}
    bound_refs = {r["event_ref"] for r in led if r["event_ref"] and r["producer"] == BINDING_PRODUCER}
    ledger_from = led[0]["at"] if led else None
    app_cache: dict[tuple, Any] = {}
    synthetic: list[tuple[datetime, float, bool]] = []   # (bid_at, intensity, graded) of first bids with no ledger row
    rows = []

    def appetite_at(snap: datetime, t: datetime):
        known = _as_of(acts, snap)
        since = t - SOURCE_HISTORY
        key = (len(known), sum(1 for k, _ in known if k.closed_at is not None),
               sum(1 for k, _ in known if (k.opened_at or k.closed_at) < since),
               sum(1 for k, _ in known if k.closed_at is not None and k.closed_at < since))
        if key not in app_cache:
            app_cache[key] = build_family_appetite(cfg, [k for k, _ in known], t)
        return app_cache[key], known

    def bid_once(t: datetime, first_bid: datetime, rule: str, s: float, *, stand_ins: bool = True) -> tuple[dict, Any]:
        """One bid of a ``rule`` ask with salience ``s`` at ``t``, as the market decides it. ``stand_ins`` False:
        the reference is the real ledger only (what the live market holds today)."""
        snap = t - snapshot_lag
        app, known = appetite_at(snap, t)
        closes = [k.closed_at for k, _ in known if k.closed_at is not None]
        ref: list[Any] = [(r["intensity"], r["graded"]) for r in led if snap - SOURCE_HISTORY <= r["at"] < snap]
        if stand_ins and (counterfactual or ledger_from is None or snap < ledger_from):
            ref += [(v, g) for bt, v, g in synthetic if snap - SOURCE_HISTORY <= bt < snap]
        event_values = (same_grading_events(ref, app.graded) if same_grading
                        else same_grading_events([v for v, _ in ref], app.graded))
        app = dataclasses.replace(app, last_close_at=max(closes) if closes else None, event_values=event_values)
        h = hunger(app, t, opened_at=first_bid)
        inten, ibasis = want_intensity(h.intensity, s)
        # As GRADES_SQL and the market read them: graded acts OPENED in the 30 days before the bid (issue #3350).
        graded = [(r, k.outcome_value) for k, r in known if k.graded and k.closed_at is not None
                  and snap - OUTCOME_WINDOW <= (k.opened_at or k.closed_at) <= snap]
        outcome = rule_outcome_values(graded)[rule]
        rank, basis, _ = rank_value(app, inten, outcome, event=True)
        return {"at": t, "rank": rank, "basis": basis, "outcome": outcome, "intensity": inten,
                "intensity_basis": ibasis, "salience": s, "n_reference": len(event_values)}, app

    for a, s in zip(asks, sal):
        if s is None or a["created_at"] < start or a["created_at"] > end:
            continue
        first_bid = a["created_at"]
        horizon = min(first_bid + LOSS_HORIZON, end)
        bids = []
        t = first_bid
        n_held = 0
        while t <= horizon:
            b, app = bid_once(t, first_bid, a["rule"], s)
            bids.append(b)
            rank, inten = b["rank"], b["intensity"]
            # A first bid with no row in the ledger stands in for one. Outside ``counterfactual`` a stand-in is counted
            # only while the market had no ledger at all (``bid_once``); once the ledger exists it is the truth.
            ref_id = f"orion_ask:{a['ask_id']}"
            if t == first_bid and ref_id not in (bound_refs if counterfactual else led_refs):
                synthetic.append((t, inten, app.graded))
            if rank > 0:
                break
            n_held += 1
            # The live wait is max(backoff, the market's own wait clamped by bind_verdict). Every replayed hold is
            # ``sated`` (no charge, asserted above), which names no wait, so bind_verdict's is REBID_DEFAULT_SEC.
            t += timedelta(seconds=rebid_after_sec(n_held, REBID_DEFAULT_SEC))
        funded_at = next((b["at"] for b in bids if b["rank"] > 0), None)
        rows.append({"ask_id": a["ask_id"], "rule": a["rule"], "answered": a["status"] == "answered",
                     "held_at_all": bids[0]["rank"] <= 0, "never_opened": funded_at is None,
                     "censored": funded_at is None and horizon < first_bid + LOSS_HORIZON,
                     "bids": len(bids), "followed_h": (horizon - first_bid).total_seconds() / 3600,
                     "first_rank": bids[0]["rank"], "first_basis": bids[0]["basis"], "outcome": bids[0]["outcome"],
                     "first_by_hunger": bids[0]["intensity_basis"] == "hunger",
                     "delay_h": None if funded_at is None else round((funded_at - first_bid).total_seconds() / 3600, 2),
                     **({"bid_log": bids} if keep_bids else {})})
    # The next bid of each rule at ``end``, against the real ledger only (what the live market holds), at salience 0
    # and 1. Rank only grows with intensity, so salience 0 is its floor and 1 its ceiling: "funded" is proved at the
    # floor, "held" at the ceiling (review finding).
    probe = {r: (bid_once(end, end, r, 0.0, stand_ins=False)[0], bid_once(end, end, r, 1.0, stand_ins=False)[0])
             for r in probe_rules}
    share = lambda xs, f: round(sum(1 for x in xs if f(x)) / len(xs), 4) if xs else None  # noqa: E731
    by_rule = {}
    for r in sorted({x["rule"] for x in rows}):
        mine = [x for x in rows if x["rule"] == r]
        late = sorted(x["delay_h"] for x in mine if x["held_at_all"] and x["delay_h"] is not None)
        by_rule[r] = {"asks": len(mine), "held_at_all": share(mine, lambda x: x["held_at_all"]),
                      "opened_later": sum(1 for x in mine if x["held_at_all"] and not x["never_opened"]),
                      "never_opened_in_horizon": share(mine, lambda x: x["never_opened"]),
                      "of_which_censored_by_replay_end": sum(1 for x in mine if x["censored"]),
                      "later_open_delay_h_median": _q(late, 0.5), "later_open_delay_h_max": late[-1] if late else None,
                      "answered": sum(1 for x in mine if x["answered"]),
                      "mean_expected_answer": round(sum(x["outcome"] for x in mine) / len(mine), 4)}
    vehicle = [x for x in rows if x["rule"] == "vehicle"]
    answerable = [x for x in rows if x["rule"] in ("confirm", "same")]
    answered = [x for x in rows if x["answered"]]
    days = max(1e-9, (end - start).total_seconds() / 86400)
    lost = [x for x in answered if x["never_opened"]]
    held = [x for x in rows if x["held_at_all"]]
    held_days = sum(x["followed_h"] if x["never_opened"] else (x["delay_h"] or 0) for x in held) / 24
    out = {
        "rule": "no_deadline (issue #3285): held re-bids 600 s doubling to 3 h until funded",
        "loss_horizon_days": LOSS_HORIZON.days,
        "asks": len(rows), "by_rule": by_rule,
        "answered_per_day": round(len(answered) / days, 3),
        "answered_per_day_under_4b": round((len(answered) - len(lost)) / days, 3),
        "answered_loss_share": share(answered, lambda x: x["never_opened"]),
        "lost_answers_by_rule": {r: sum(1 for x in lost if x["rule"] == r) for r in sorted({x["rule"] for x in lost})},
        # want_ledger volume: a held candidate's re-bids (the 3 h backoff cap keeps this near 8 a day).
        "rebids_per_held_candidate_day": round(sum(x["bids"] - 1 for x in held) / held_days, 2) if held_days else None,
        # Hunger counts from the family's last ask close (4a gives it acts), not from the want's opened_at:
        # the share of first bids whose intensity came from hunger rather than salience (review finding).
        "first_bid_intensity_from_hunger_share": share(rows, lambda x: x["first_by_hunger"]),
        "graded_basis_share": share(rows, lambda x: x["first_basis"].endswith("graded") and "ungraded" not in x["first_basis"]),
        "event_rank_rule": "same_grading (issue #3364)" if same_grading else "every past event (before #3364)",
        "ledger_rows": len(led),
        "next_bid_by_rule": {r: {"rank_floor": round(lo["rank"], 4), "rank_ceiling": round(hi["rank"], 4),
                                 "funded_at_any_salience": lo["rank"] > 0, "held_at_any_salience": hi["rank"] <= 0,
                                 "expected_answer": round(lo["outcome"], 4), "basis": hi["basis"],
                                 "reference_n": lo["n_reference"]} for r, (lo, hi) in probe.items()},
    }
    if keep_bids:
        out["rows"] = rows
    out["checks"] = {
        "vehicle_held_at_least_70pct": (share(vehicle, lambda x: x["held_at_all"]) or 0) >= GATE_VEHICLE_HELD_MIN
        if vehicle else False,
        # Held must mean held, not "held once, opened 10 minutes later" (review finding): the same share of vehicle
        # asks must also stay unopened through the horizon.
        "vehicle_never_opened_in_horizon_at_least_70pct": (share(vehicle, lambda x: x["never_opened"]) or 0)
        >= GATE_VEHICLE_HELD_MIN if vehicle else False,
        "confirm_and_same_held_at_most_10pct": (share(answerable, lambda x: x["held_at_all"]) or 0) <= GATE_ANSWERABLE_HELD_MAX
        if answerable else False,
        "answered_per_day_falls_at_most_10pct": (out["answered_loss_share"] or 0) <= GATE_ANSWERED_LOSS_MAX,
    }
    return out


def _camera_fixture() -> int:
    """Deterministic: 30 days of asks with the live answer pattern (vehicles mostly "don't know",
    confirm/same nearly always answered). Checks the replay's mechanics, not the live verdict."""
    fams = parse_families({"version": 1, "families": {ASK_FAMILY: {"tier": 2, "deferrable_window_s": 86400, "lanes": []}}})
    t0 = datetime(2026, 10, 1, 7, tzinfo=timezone.utc)
    asks = []
    for d in range(60):
        for j, rule in enumerate(("confirm", "confirm", "same", "vehicle")):
            at = t0 + timedelta(days=d, minutes=10 * j)
            vehicle = rule == "vehicle"
            answered = not vehicle or d % 8 == 0
            asks.append({"ask_id": f"{rule}{d}-{j}", "rule": rule, "created_at": at, "expires_at": at + timedelta(days=7),
                         "status": "answered" if answered else "dismissed",
                         "answered_at": at + timedelta(hours=4), "answer_detail": None if answered else {"dont_know": True},
                         # Vehicles are seen most (the backwards value); people less.
                         "sightings": 40 + d % 7 if vehicle else 3 + (d + j) % 9})
    start = t0 + timedelta(days=30)
    r = camera_asks_gate(fams, asks, start=start, end=t0 + timedelta(days=60))
    br = r["by_rule"]
    checks = {
        "family_ranks_graded": (r["graded_basis_share"] or 0) == 1.0,
        "vehicle_expected_answer_is_low": br["vehicle"]["mean_expected_answer"] < 0.3,
        "confirm_expected_answer_is_high": br["confirm"]["mean_expected_answer"] > 0.9,
        "vehicles_held_more_than_confirms": (br["vehicle"]["held_at_all"] or 0) > (br["confirm"]["held_at_all"] or 0),
    }
    print(json.dumps({"lane": "camera_asks_fixture", **r, "fixture_checks": checks}, indent=2, default=str))
    return 0 if all(checks.values()) else 1


BINDING_PRODUCER = "orion-sql-writer.ask_events.binding"
PARITY_TOL = 1e-6          # the ledger stores value_rank rounded to 6 places


def live_parity(families: FamiliesConfig, asks: Sequence[Mapping[str, Any]], ledger: Sequence[Mapping[str, Any]],
                *, same_grading: bool, snapshot_lag: timedelta = MARKET_SNAPSHOT_LAG,
                start: Optional[datetime] = None) -> dict:
    """Re-decide every live 4b bid (``ledger`` rows of ``BINDING_PRODUCER``) with this replay and compare it with
    what the market decided. ``same_grading`` False is the rule the market ran when those bids were made, so it
    must reproduce them (issue #3364); True is the rule after the fix. Pure."""
    live = [r for r in ledger if r.get("producer") == BINDING_PRODUCER and r.get("value_rank") is not None]
    by_ref = {f"orion_ask:{a['ask_id']}": a for a in asks}
    if not live:
        return {"bids": 0, "matched": 0, "rows": []}
    t0 = min(_ts(r["decided_at"]) for r in live)
    t1 = max(_ts(r["decided_at"]) for r in live)
    out = camera_asks_gate(families, asks, start=start or t0 - timedelta(minutes=10), end=t1 + timedelta(seconds=1),
                           ledger=ledger, same_grading=same_grading, snapshot_lag=snapshot_lag, keep_bids=True)
    first = {f"orion_ask:{x['ask_id']}": x["bid_log"][0] for x in out["rows"]}
    rows = []
    for r in sorted(live, key=lambda r: _ts(r["decided_at"])):
        ref = r.get("event_ref")
        b = first.get(ref)
        ask = by_ref.get(ref)
        live_funded = r.get("outcome") == "funded"
        row = {"event_ref": ref, "rule": ask["rule"] if ask else None, "decided_at": str(r["decided_at"]),
               "live_rank": round(float(r["value_rank"]), 4), "live_funded": live_funded,
               "replay_rank": None if b is None else round(b["rank"], 4),
               "replay_funded": None if b is None else b["rank"] > 0,
               "replay_outcome": None if b is None else round(b["outcome"], 4),
               "replay_reference_n": None if b is None else b["n_reference"]}
        row["match"] = (b is not None and abs(b["rank"] - float(r["value_rank"])) <= PARITY_TOL
                        and (b["rank"] > 0) == live_funded)
        rows.append(row)
    return {"bids": len(rows), "matched": sum(1 for x in rows if x["match"]),
            "held": sum(1 for x in rows if x["replay_funded"] is False), "rows": rows}


ASKS_FOR_REPLAY_SQL = """select coalesce(json_agg(r order by r.created_at), '[]'::json) from (
        select a.ask_id, a.source_kind, a.evidence_refs, a.status, a.answer_detail, a.created_at, a.expires_at,
               a.answered_at, (select i.kind from vision_individual i
                                where i.individual_id = split_part(a.source_ref, '|', 1)) as individual_kind,
               (select count(*) from vision_individual_sighting v where v.individual_id = split_part(a.source_ref, '|', 1)
                  and v.started_at <= a.created_at and not v.stationary) as sightings
          from orion_ask a
         where {where} and a.created_at >= '{since}') r;"""

# The family's event history exactly as the market reads it (orion-durable-runs app/physiology_market.py
# EVENT_HISTORY_SQL: first decision per want, intensity and grading), with what this replay compares.
LEDGER_FOR_REPLAY_SQL = """select coalesce(json_agg(r order by r.decided_at), '[]'::json) from (
        select distinct on (want_id) want_id, producer, decided_at, outcome, reason,
               (decision->>'intensity')::float as intensity, (decision->>'value_rank')::float as value_rank,
               case jsonb_typeof(decision->'graded') when 'boolean' then (decision->>'graded')::boolean end as graded,
               decision->'want'->>'event_ref' as event_ref
          from want_ledger
         where family = '{family}' and decided_at >= '{since}' and decision->'want' ? 'salience'
           and decision->>'intensity' is not null and not (producer = any(array[{synthetic}]))
         order by want_id, decided_at) r;"""


def _read_camera_inputs(since: datetime) -> tuple[list[dict], list[dict]]:
    import _body_economy_data as data_mod

    from orion.grammar.camera_ask_acts import ORION_MADE_CAMERA_ASK_SQL, ask_rule
    from orion.physiology.choice import SYNTHETIC_PRODUCERS

    iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = data_mod.run_query(ASKS_FOR_REPLAY_SQL.format(where=ORION_MADE_CAMERA_ASK_SQL, since=iso))
    asks = [{**r, "created_at": _ts(r["created_at"]), "expires_at": _ts(r.get("expires_at")),
             "answered_at": _ts(r.get("answered_at")),
             "rule": ask_rule(r["source_kind"], r.get("evidence_refs"), r.get("individual_kind"))} for r in rows]
    synthetic = ", ".join(f"'{p}'" for p in sorted(SYNTHETIC_PRODUCERS)) or "''"
    ledger = data_mod.run_query(LEDGER_FOR_REPLAY_SQL.format(family=ASK_FAMILY, since=iso, synthetic=synthetic))
    return asks, ledger


def _camera_live(days: int, out_json: Optional[str], parity_fixture: Optional[str] = None) -> int:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    asks, ledger = _read_camera_inputs(start - SOURCE_HISTORY - timedelta(days=1))
    families, _ = load_families(ROOT / "config" / "physiology" / "families.yaml")
    from orion.grammar.camera_ask_acts import RULES

    # The gate: 4b's rule as if it had bid every ask of the window (stand-ins for asks with no 4b bid).
    r = camera_asks_gate(families, asks, start=start, end=end, ledger=ledger, probe_rules=RULES, counterfactual=True)
    # What the market really had (the ledger as it was), under the rule after and before #3364.
    as_ledger = camera_asks_gate(families, asks, start=start, end=end, ledger=ledger)
    before = camera_asks_gate(families, asks, start=start, end=end, ledger=ledger, same_grading=False, probe_rules=RULES)
    # Parity over the whole window: the full replay, stand-ins and all, must reproduce every live bid.
    parity_old = live_parity(families, asks, ledger, same_grading=False, start=start)
    parity_new = live_parity(families, asks, ledger, same_grading=True, start=start)
    out = {"lane": "camera_asks_live", "window_days": days, "as_of": end.isoformat(), **r,
           "as_ledger_same_grading": {"by_rule": as_ledger["by_rule"], "checks": as_ledger["checks"],
                                      "answered_loss_share": as_ledger["answered_loss_share"]},
           "rule_before_3364": {"by_rule": before["by_rule"], "checks": before["checks"],
                                "answered_loss_share": before["answered_loss_share"],
                                "next_bid_by_rule": before["next_bid_by_rule"]},
           "live_parity_rule_live_then": parity_old, "live_bids_under_same_grading": parity_new}
    nb = r["next_bid_by_rule"]
    out["checks"] = {**r["checks"],
                     # The replay must reproduce every live 4b decision under the rule the market ran then.
                     "replay_reproduces_live_bids": parity_old["bids"] > 0 and parity_old["matched"] == parity_old["bids"],
                     # What the live market decides for the next ask, from today's ledger (issue #3364).
                     "next_vehicle_bid_held": nb["vehicle"]["held_at_any_salience"],
                     "next_confirm_and_same_bids_funded": (nb["confirm"]["funded_at_any_salience"]
                                                           and nb["same"]["funded_at_any_salience"])}
    text = json.dumps(out, indent=2, default=str)
    if out_json:
        Path(out_json).write_text(text)
    print(text)
    if parity_fixture:
        fx = export_parity_fixture(asks, ledger)
        # One row per line: a small, reviewable diff when the fixture is refreshed.
        lines = ["{"] + [f'"{k}": [' + ",\n".join(json.dumps(r, default=str, sort_keys=True) for r in v)
                         + ("]," if k == "asks" else "]") for k, v in fx.items()] + ["}"]
        Path(parity_fixture).write_text("\n".join(lines) + "\n")
    return 0 if all(out["checks"].values()) else 1


def export_parity_fixture(asks: Sequence[Mapping[str, Any]], ledger: Sequence[Mapping[str, Any]]) -> dict:
    """The live inputs ``live_parity`` needs, with nothing personal: ids hashed, answer_detail reduced to the two
    flags ``ask_grade`` reads, no names, no individual ids, no evidence refs."""
    import hashlib

    def h(x: Any) -> str:
        return "ask-" + hashlib.sha256(str(x).encode()).hexdigest()[:16]

    def ref(v: Optional[str]) -> Optional[str]:
        return f"orion_ask:{h(v.split(':', 1)[1])}" if isinstance(v, str) and v.startswith("orion_ask:") else None

    def detail(d: Any) -> dict:
        d = d if isinstance(d, dict) else (json.loads(d) if isinstance(d, str) and d.strip().startswith("{") else {})
        # ask_grade reads ``"resolved_by" in detail`` (presence) and ``bool(detail.get("dont_know"))``.
        return {**({"resolved_by": True} if "resolved_by" in d else {}), **({"dont_know": True} if d.get("dont_know") else {})}

    return {
        "asks": [{"ask_id": h(a["ask_id"]), "rule": a["rule"], "status": a["status"], "sightings": int(a["sightings"] or 0),
                  "created_at": a["created_at"], "expires_at": a.get("expires_at"), "answered_at": a.get("answered_at"),
                  "answer_detail": detail(a.get("answer_detail"))} for a in asks],
        "ledger": [{"producer": r["producer"], "decided_at": r["decided_at"], "outcome": r["outcome"],
                    "intensity": r["intensity"], "value_rank": r["value_rank"], "graded": r["graded"],
                    "event_ref": ref(r.get("event_ref"))} for r in ledger],
    }


# --- live lane -----------------------------------------------------------------------------------

def _live(days: int, out_json: Optional[str]) -> int:
    import _body_economy_data as data_mod

    from orion.grammar.camera_ask_acts import ORION_MADE_CAMERA_ASK_SQL   # Juniper's own cards are not asks

    q = data_mod.run_query
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    hs = start - SOURCE_HISTORY - timedelta(days=1)
    iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    reads = q(f"""select coalesce(json_agg(r order by r.at), '[]'::json) from (
        select s.seed_id, s.stage2_completed_at as at,
               (select count(*) from jsonb_array_elements(case when jsonb_typeof(s.stage2_result_json->'relationship_claims') = 'array'
                    then s.stage2_result_json->'relationship_claims' else '[]'::jsonb end) c
                 where c->'receipt'->>'outcome' = 'accepted_provisional') as links
          from world_pulse_read_seed s
         where s.stage2_status = 'done' and s.stage2_completed_at >= '{iso(hs)}') r;""")
    asks = q(f"""select coalesce(json_agg(r order by r.at), '[]'::json) from (
        select a.ask_id, a.created_at as at, a.answered_at, a.expires_at, a.status,
               (select count(*) from vision_individual_sighting v where v.individual_id = split_part(a.source_ref, '|', 1)
                  and v.started_at <= a.created_at and not v.stationary) as sightings
          from orion_ask a
         where {ORION_MADE_CAMERA_ASK_SQL}
           and a.created_at >= '{iso(hs)}') r;""")
    acts_rows = q(f"""select coalesce(json_agg(r), '[]'::json) from (
        select act_id, actor, opened_at, closed_at, outcome_value, outcome_source
          from grammar_acts where actor in ('{READ_FAMILY}', '{ASK_FAMILY}')
           and coalesce(opened_at, closed_at, created_at) >= '{iso(hs - SOURCE_HISTORY)}') r;""")
    prime_rows = q(f"""select coalesce(json_agg(r), '[]'::json) from (
        select verb, opened_at from grammar_acts
         where actor = 'prime' and verb in ('infer', 'predict') and opened_at >= '{iso(start)}') r;""")
    events: list[Event] = []
    for r in reads:
        events.append(Event(family=READ_FAMILY, ref=f"world_pulse_read_seed:{r['seed_id']}", at=_ts(r["at"]),
                            magnitude=int(r["links"] or 0)))
    for r in asks:
        at = _ts(r["at"])
        answered, expires = _ts(r.get("answered_at")), _ts(r.get("expires_at"))
        # Answered: gone then. Dismissed/withdrawn/expired rows carry no close time, so expiry is the
        # bound (this can overstate Prime's coverage a little).
        ends = answered or expires
        events.append(Event(family=ASK_FAMILY, ref=f"orion_ask:{r['ask_id']}", at=at, magnitude=int(r["sightings"] or 0),
                            ends_at=ends, posted=ends is None or ends > at + POLL_DELAY))
    acts = [ActRecord(act_id=str(r["act_id"]), family=str(r["actor"]), opened_at=_ts(r.get("opened_at")),
                      closed_at=_ts(r.get("closed_at")),
                      outcome_value=float(r["outcome_value"]) if r.get("outcome_value") is not None else None,
                      outcome_source=r.get("outcome_source")) for r in acts_rows]
    thoughts = sorted(_ts(r["opened_at"]) for r in prime_rows if r["verb"] == "infer")
    bets = sorted(_ts(r["opened_at"]) for r in prime_rows if r["verb"] == "predict")
    families, _ = load_families(ROOT / "config" / "physiology" / "families.yaml")
    r = report(families, events, acts, start=start, thoughts=thoughts, concern_bets=bets)
    out = {"lane": "live", "window_days": days, "as_of": end.isoformat(), **r}
    text = json.dumps(out, indent=2, default=str)
    if out_json:
        Path(out_json).write_text(text)
    print(text)
    return 0 if all(r["checks"].values()) else 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out-json")
    ap.add_argument("--camera-asks", action="store_true", help="the 4b gate for camera asks (patch 4a)")
    ap.add_argument("--export-parity-fixture", help="with --camera-asks --live: write the live inputs, anonymised")
    a = ap.parse_args(argv)
    if a.camera_asks:
        return _camera_live(a.days, a.out_json, a.export_parity_fixture) if a.live else _camera_fixture()
    return _live(a.days, a.out_json) if a.live else _fixture()


if __name__ == "__main__":
    raise SystemExit(main())
