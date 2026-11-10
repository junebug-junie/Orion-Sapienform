#!/usr/bin/env python3
"""Outreach binding replay (READ-ONLY): what Patch D for hub.outreach would have done to each real message.

Body economy Patch D for Hub outreach (services/orion-hub/scripts/outreach_bind.py). Every real outreach
attempt that reached its COMPOSE point in the window (``endogenous_outreach_decisions``: sent, Orion's PASS,
an empty or refused generation, a drop after generation) from a door that now binds -- the background
tick, reading findings, self findings, office visits -- is re-bid with the live Choice code
(``orion.physiology.choice.decide``) against the ``physiology_state`` row in force, and the verdict goes
through the same pure rule Hub obeys (``orion.physiology.binding.bind_verdict``). A held want re-bids at
the verdict's ``retry_at`` until it is admitted, never past its 1 h window (``window_end``).

Per attempt: let through at its first bid, or held and then admitted (funded later, queued at the pool,
or at the window's end), and how long. Curiosity Door-A, the strain report and forced sends do not bid
(see outreach_bind.py), so they are not replayed.

Approximations (said in the report): the attempt's time is its shadow bid's time (the tick's compose point,
``want_ledger`` row with ``want_id`` = the attempt's correlation id) when there is one, else the decision
row's time (minutes after the compose point for a sent message). Act history and lane levels are the REAL
ones: a held message would have closed its act later, so hunger after a hold reads lower than it would
have. A funded want's reservation is released at once (an outreach act is short), and other families'
reservations made inside the window are not added to the book, so capacity deferrals are under-counted
(a capacity deferral worth its price is admitted anyway, so this cannot hide a hold).

Writes nothing to any database.

    python3 scripts/analysis/outreach_binding_replay.py --days 14 --out-json /tmp/x.json --out-md /tmp/x.md \
        [--cache /tmp/x.cache.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

WINDOW_SEC = 3600.0                     # config/physiology/families.yaml hub.outreach deferrable_window_s
TICK = "endogenous_tick"
# Doors that bind besides the tick (services/orion-hub/scripts/outreach_bind.py). The tick's rows carry no
# source. self_finding_outreach bound until it was retired (2026-11-10); its past rows still replay. Since
# 2026-11-10 (one attention budget) every door is cap-counted, so every attempt replays cap-counted.
DOOR_SOURCES = ("reading_finding_outreach", "self_finding_outreach", "office_reach_out")
BINDING_DOORS = (TICK,) + DOOR_SOURCES
# Decision reasons that mean the attempt reached its compose point (the bid happens just before it).
COMPOSED_REASONS = ("sent", "orion_passed", "empty_generation", "named_ungrounded_signal", "no_question",
                    "claimed_recognition")
MAX_BIDS = 1 + int(WINDOW_SEC // 120.0) + 1   # one bid per relief window of at least 2 min, plus the first


def _ts(v: Any) -> Optional[datetime]:
    if v is None or isinstance(v, datetime):
        return v
    s = str(v).replace("Z", "+00:00")
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def door_of(source: Optional[str]) -> Optional[str]:
    """The binding door a decision row belongs to, or None (Door-A, strain report, anything else)."""
    s = (source or "").strip()
    if not s:
        return TICK
    return s if s in DOOR_SOURCES else None


def is_composed(reason: str) -> bool:
    return reason in COMPOSED_REASONS or reason.endswith("_after_generation")


def attempts_from_rows(rows: list[dict], ledger_times: dict[str, datetime]) -> list[dict]:
    """Real compose attempts of binding doors, time-ordered. Pure."""
    out = []
    for r in rows:
        if r.get("forced"):
            continue
        door = door_of(r.get("source"))
        reason = str(r.get("reason") or "")
        if door is None or not is_composed(reason):
            continue
        t = _ts(r.get("decided_at"))
        corr = r.get("correlation_id")
        lt = ledger_times.get(str(corr)) if corr else None
        if lt is not None and t is not None and lt <= t:
            t = lt
        if t is None:
            continue
        out.append({"decision_id": r.get("decision_id"), "door": door, "at": t, "reason": reason,
                    "correlation_id": corr, "cap_counted": True})
    return sorted(out, key=lambda a: a["at"])


Decider = Callable[[dict, datetime, int], Any]


def replay_attempt(attempt: dict, decide_at: Decider, *, data_end: Optional[datetime] = None) -> dict:
    """Bid, obey, re-bid at ``retry_at`` until admitted. ``decide_at(attempt, t, n)`` is the market's
    answer to the n-th bid at time t (a WantDecisionV1-like object, or None for no answer). Pure."""
    from orion.physiology.binding import bind_verdict

    t0 = attempt["at"]
    end = t0 + timedelta(seconds=WINDOW_SEC)
    t, bids = t0, []
    verdict = None
    for n in range(MAX_BIDS):
        if data_end is not None and t > data_end:
            verdict = "censored_at_data_end"
            break
        if t >= end:
            bids.append({"at": t.isoformat(), "outcome": None, "reason": None, "binding": None, "net": None,
                         "verdict": "window_end"})
            verdict = "window_end"
            break
        d = decide_at(attempt, t, n)
        v = bind_verdict(d, t, end)
        bids.append({"at": t.isoformat(), "outcome": getattr(d, "outcome", None), "reason": getattr(d, "reason", None),
                     "binding": getattr(d, "binding_constraint", None), "net": getattr(d, "net", None),
                     "verdict": v.why})
        if v.admit:
            verdict = v.why
            break
        t = v.retry_at
    held = (t - t0).total_seconds() if verdict and verdict != "censored_at_data_end" else None
    return {"decision_id": attempt["decision_id"], "door": attempt["door"], "at": t0.isoformat(),
            "real_reason": attempt["reason"], "verdict": verdict or "never_decided", "n_bids": len(bids),
            "held_sec": round(held, 1) if held is not None else None, "bids": bids}


def summarize(rows: list[dict]) -> dict:
    def q(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        return round(s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))], 1)

    by_door: dict[str, dict] = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["door"]].append(r)
    groups["all"] = list(rows)
    for door, rs in groups.items():
        held = [r["held_sec"] for r in rs if (r["held_sec"] or 0) > 0]
        first = Counter(f"{r['bids'][0]['outcome']}:{r['bids'][0]['reason']}" for r in rs if r["bids"])
        by_door[door] = {
            "attempts": len(rs), "sent_really": sum(1 for r in rs if r["real_reason"] == "sent"),
            "let_through_at_first_bid": sum(1 for r in rs if r["n_bids"] == 1 and (r["held_sec"] or 0) == 0),
            "held_then_admitted": len(held),
            "held_sends": sum(1 for r in rs if (r["held_sec"] or 0) > 0 and r["real_reason"] == "sent"),
            "verdicts": dict(Counter(r["verdict"] for r in rs)),
            "first_bid": dict(first),
            "held_sec_median": q(held, 0.5), "held_sec_p90": q(held, 0.9), "held_sec_max": max(held) if held else None,
            "held_to_window_end": sum(1 for r in rs if r["verdict"] == "window_end"),
            "held_over_window": sum(1 for r in rs if (r["held_sec"] or 0) > WINDOW_SEC + 1),
            "bids_total": sum(r["n_bids"] for r in rs),
            "deferral_reasons_all_bids": dict(Counter(b["reason"] for r in rs for b in r["bids"]
                                                      if b["outcome"] == "deferred")),
        }
    return by_door


# --- live data ------------------------------------------------------------------------------------

def fetch_rows(start: datetime, end: datetime) -> dict:
    import body_economy_shadow_vs_real as svr

    s, e = svr._iso(start), svr._iso(end)
    return {
        "decisions": svr._q(f"""select coalesce(json_agg(r order by r.decided_at), '[]'::json) from (
            select decision_id, decided_at, reason, forced, correlation_id, result_json->>'source' as source
              from endogenous_outreach_decisions where decided_at >= '{s}' and decided_at < '{e}'
               and outreach is not null) r;"""),
        "outreach_ledger": svr._q(f"""select coalesce(json_agg(r), '[]'::json) from (
            select want_id, min(decided_at) as decided_at from want_ledger
             where family = 'hub.outreach' and decided_at >= '{s}' and decided_at < '{e}' group by want_id) r;"""),
    }


def live_decider(ex: dict) -> Decider:
    """The live Choice code against the stored body state in force at each bid."""
    import body_economy_shadow_vs_real as svr

    from orion.physiology.appetite import ActRecord
    from orion.physiology.choice import FundedWant, ReservationBook, SYNTHETIC_PRODUCERS, build_context, decide
    from orion.physiology.weights import UseRecord, weight_series
    from orion.schemas.grammar import ActCostV2
    from orion.schemas.physiology import PhysiologyStateV1
    from orion.schemas.physiology_want import WantV1, WantWindowV1

    ts = _ts
    fams = svr._families()
    acts = [ActRecord(act_id=str(r["act_id"]), family=str(r["actor"]), opened_at=ts(r.get("opened_at")),
                      closed_at=ts(r.get("closed_at")), outcome_value=r.get("outcome_value"),
                      outcome_source=r.get("outcome_source"), signal_id=r.get("signal_id"),
                      cost=r.get("cost_json") if isinstance(r.get("cost_json"), dict) else {}) for r in ex["acts"]]
    uses = [UseRecord(act_id=str(r["act_id"]), family=str(r["actor"]), started_at=ts(r.get("started_at")),
                      link_status=str(r["link_status"]), used=r.get("used"), window_closed=bool(r.get("window_closed")))
            for r in ex["uses"]]
    states = [PhysiologyStateV1.model_validate(s) for s in ex["states"]]
    start, end = ts(ex["start"]), ts(ex["end"])
    series = weight_series(fams, acts, uses, start=(start - timedelta(days=1)).date(), end=end.date())
    book = ReservationBook()
    book.seed_funded([FundedWant(str(r["decision_id"]), ts(r["decided_at"]), str(r["family"]), int(r["tier"]))
                      for r in ex.get("ledger_funded_before") or [] if r.get("producer") not in SYNTHETIC_PRODUCERS],
                     start)
    cache: dict = {}

    def state_at(t):
        cur = None
        for s in states:          # states are time-ordered
            if s.generated_at <= t:
                cur = s
            else:
                break
        return cur

    def decide_at(attempt: dict, t: datetime, n: int):
        st = state_at(t)
        key = (st.tick_id if st is not None else None, t.replace(second=0, microsecond=0))
        if key not in cache:
            cache.clear()
            before = [a for a in acts if (a.closed_at or a.opened_at) is not None and (a.closed_at or a.opened_at) < t]
            cache[key] = build_context(fams, before, [u for u in uses if u.started_at and u.started_at < t], st, t,
                                       weights=series.get(t.date(), {}))
        want_id = f"replay-{attempt['decision_id']}"
        want = WantV1(want_id=want_id, family="hub.outreach", producer="replay", verb="say", tier=2,
                      object_ref="juniper", correlation_id=want_id, opened_at=attempt["at"],
                      window=WantWindowV1(latest=attempt["at"] + timedelta(seconds=WINDOW_SEC)),
                      cost=None if attempt["cap_counted"] else ActCostV2(juniper_msgs=0))
        d = decide(want, st, cache[key], book, t)
        book.release(want_id)     # an outreach act is short; keep the book honest
        return d

    return decide_at


def run(ex: dict, decide_at: Optional[Decider] = None) -> dict:
    ledger_times = {str(r["want_id"]): _ts(r["decided_at"]) for r in ex.get("outreach_ledger") or []}
    attempts = attempts_from_rows(ex["decisions"], ledger_times)
    decide_at = decide_at or live_decider(ex)
    data_end = _ts(ex["end"])
    rows = [replay_attempt(a, decide_at, data_end=data_end) for a in attempts]
    return {"start": ex["start"], "end": ex["end"], "summary": summarize(rows), "rows": rows}


def render_md(report: dict) -> str:
    L = [f"# Outreach binding replay (hub.outreach), {report['start'][:16]}Z .. {report['end'][:16]}Z", "",
         "Each real compose attempt of a binding door, re-bid with the live market and obeyed "
         "(scripts/analysis/outreach_binding_replay.py). Approximations in the module docstring.", ""]
    for door, s in report["summary"].items():
        L += [f"## {door}", "",
              f"{s['attempts']} attempts ({s['sent_really']} really sent). Let through at the first bid: "
              f"{s['let_through_at_first_bid']}. Held then admitted: {s['held_then_admitted']} "
              f"({s['held_sends']} of them real sends), {s['held_to_window_end']} at the 1 h window's end, "
              f"{s['held_over_window']} past it.",
              f"Hold: median {s['held_sec_median']} s, p90 {s['held_sec_p90']} s, max {s['held_sec_max']} s.",
              f"Verdicts: {s['verdicts']}. First bid: {s['first_bid']}. {s['bids_total']} bids in all; "
              f"deferral reasons over all bids: {s['deferral_reasons_all_bids']}.", ""]
    held = [r for r in report["rows"] if (r["held_sec"] or 0) > 0]
    if held:
        L += ["## Held attempts", "", "| at | door | real outcome | verdict | bids | held (s) |", "|---|---|---|---|---|---|"]
        for r in held:
            L.append(f"| {r['at'][:16]} | {r['door']} | {r['real_reason']} | {r['verdict']} | {r['n_bids']} | {r['held_sec']} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    import body_economy_shadow_vs_real as svr

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--end", default=None, help="ISO time; default now")
    ap.add_argument("--cache", default=None, help="read/write the extract as JSON")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args(argv)
    end = _ts(a.end) if a.end else datetime.now(timezone.utc).replace(microsecond=0)
    if a.cache and Path(a.cache).exists():
        ex = json.loads(Path(a.cache).read_text())
    else:
        start = end - timedelta(days=a.days)
        ex = svr.fetch(start, end)
        ex.update(fetch_rows(start, end))
        if a.cache:
            Path(a.cache).write_text(json.dumps(ex, default=str))
    report = run(ex)
    Path(a.out_json).write_text(json.dumps(report, indent=1, default=str))
    md = render_md(report)
    Path(a.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
