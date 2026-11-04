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
   with ``event=True``) against the source's earlier event wants. Both families have no lanes and no fixed
   cost, so no charge applies and Choice funds every want whose rank is above 0 (asserted, not assumed).
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

    PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/event_wants_replay.py [--live --days 14]
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


def simulate(families: FamiliesConfig, events: list[Event], acts: Sequence[ActRecord]) -> list[Event]:
    """Salience, then the market's event-want decision, in time order. Mutates and returns ``events``."""
    for fam in {e.family for e in events}:
        cfg = families.get(fam)
        assert cfg is not None, fam
        # No lanes and no fixed cost: Choice charges nothing, so it funds iff the rank is above 0.
        assert not cfg.lanes and not cfg.fixed_cost, f"{fam} has a charge; this replay assumes none"
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
        }
    out["prime_thoughts"] = prime_cover(events, thoughts)
    out["prime_concern_bets"] = prime_cover(events, concern_bets)
    src = out["sources"]
    checks = {f"{fam}:most_events_low": v["salience_all_events"].get("share_under_0_5", 0) > 0.5 for fam, v in src.items()}
    checks.update({f"{fam}:event_rank_separates_better": (v["separation_new"] or 0) > (v["separation_old"] or 0)
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
    r = report(fams, events, acts, start=start, thoughts=thoughts, concern_bets=thoughts[::10])
    s = r["sources"][READ_FAMILY]
    checks = {
        "rest_point_is_zero": s["salience_all_events"]["share_zero"] > 0.75,
        "most_events_low": r["checks"][f"{READ_FAMILY}:most_events_low"],
        "very_salient_rank_not_saturated": s["very_salient_rank_at_least_0_95_new"] < s["very_salient_rank_at_least_0_95_old"],
        "prime_not_starved": r["diagnostics"]["thoughts_with_an_event"] >= COVER_MIN,
        "a_gap_starves": prime_cover(events, [t0 + timedelta(days=200)])["share_with_an_event"] == 0.0,
    }
    print(json.dumps({"lane": "fixture", **r, "checks": checks}, indent=2, default=str))
    return 0 if all(checks.values()) else 1


# --- live lane -----------------------------------------------------------------------------------

def _live(days: int, out_json: Optional[str]) -> int:
    import _body_economy_data as data_mod

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
         where a.source_kind in ('vision_individual', 'vision_person_confirm', 'vision_same_individual')
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
        ends = answered or (expires if r["status"] != "open" else None) or expires
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
    a = ap.parse_args(argv)
    return _live(a.days, a.out_json) if a.live else _fixture()


if __name__ == "__main__":
    raise SystemExit(main())
