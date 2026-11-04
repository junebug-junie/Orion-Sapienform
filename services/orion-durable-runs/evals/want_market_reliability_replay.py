#!/usr/bin/env python3
"""Eval: what would the want market have chosen with the self-monitor's track record as evidence? READ-ONLY.

Attention as the want competition, patch 1 (docs/superpowers/specs/2026-11-04-attention-as-want-
competition-design.md). Patch 1 adds weights source (d), the self-monitor's reliability per family
(``orion.physiology.weights``, ``reliability=True``). This replays the last ``--days`` of REAL
``want_ledger`` decisions with the family weight recomputed two ways, day by day, from the same 30-day
act and use history the market reads:

* **off**: sources (a) and (b) only, as live today.
* **on**: sources (a), (b) and (d).

The recorded weight cannot be recomputed exactly from today's tables: use windows that were open then
are closed now, closes arrived later, and families.yaml gained families (normalization changed). So
the counterfactual weight is the RECORDED weight plus what source (d) changes on the same history:
``weight' = recorded + (on - off)``. How often off matches the recorded weight is reported as a
diagnostic (``repro_share``), not a gate. Acceptance check 6 (with the source off, decisions equal
today's) is a unit test instead: ``reliability`` defaults off and the live caller does not pass it.

Each decision's own ``value_rank`` and ``charge`` are held as recorded; only the weight changes, so
``net' = weight' x value_rank - charge``. A decision whose outcome did not depend on the weight
(``capacity_reserved``, ``no_acceptable_lane``, ``physiology_unavailable``, ...) keeps its outcome. An
``exploration_floor`` fund stays funded (the floor funds on price, not on weight). Reservations are not
re-simulated: a flip here changes who holds a lane later, which this replay does not follow. It is a
first-order answer to "who would gain or lose funding", not a full market re-run.

Event wants (``salience``) have no history yet: no producer posts them before patch 2, so this replay
cannot show them, and says so.

Two lanes:

* ``--fixture`` (default; deterministic, no database): a reliable and an unreliable family with equal
  priors. Gates: off reproduces the recorded weights; on raises the reliable family and lowers the other,
  each within the drift cap; with no graded closes on equals off.
* ``--live``: the real ledger and act history. Gates: no Tier 2 family that had funded wants drops under the 5%
  exploration floor's share (the floor would fund it anyway, but a drop under it is a starvation flag).

    PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/want_market_reliability_replay.py [--live --days 14]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC), str(ROOT / "scripts" / "analysis")]

from orion.physiology.appetite import ActRecord  # noqa: E402
from orion.physiology.families import FamiliesConfig, load_families, parse_families  # noqa: E402
from orion.physiology.weights import (  # noqa: E402
    EVIDENCE_WINDOW,
    MAX_DRIFT_PER_DAY,
    UseRecord,
    max_daily_move,
    weight_series,
    weights_now,
)

WEIGHT_DEPENDENT = frozenset({"net_positive", "price", "sated", "exploration_floor"})
FLOOR_SHARE = 0.05
REPRO_TOL = 0.02


def _ts(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def replay(families: FamiliesConfig, decisions: Sequence[Mapping[str, Any]], acts: Sequence[ActRecord],
           uses: Sequence[UseRecord]) -> dict:
    """Pure core. ``decisions``: want_ledger rows with decided_at, family, tier, outcome, reason,
    family_weight, value_rank, charge."""
    if not decisions:
        return {"decisions": 0, "families": {}, "repro_share": None, "starved": [], "max_daily_move_on": 0.0}
    days = sorted({_ts(d["decided_at"]).date() for d in decisions})
    start, end = days[0], days[-1]
    # Exactly what the market computes on each day (weights_now replays the 30 days ending that day),
    # with acts known by that day's end only, so a later close never leaks into an earlier weight.
    off: dict[date, dict] = {}
    on: dict[date, dict] = {}
    for day in days:
        cut = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        seen = [a for a in acts if (a.closed_at or a.opened_at) is not None and (a.closed_at or a.opened_at) < cut]
        noon = cut - timedelta(hours=12)
        off[day] = weights_now(families, seen, uses, noon)
        on[day] = weights_now(families, seen, uses, noon, reliability=True)
    fam: dict[str, dict] = defaultdict(lambda: {"n": 0, "weight_dependent": 0, "funded_before": 0, "funded_after": 0,
                                                "gained": 0, "lost": 0, "w_ledger": [], "w_off": [], "w_on": [], "w_new": []})
    repro_ok = repro_n = 0
    for d in decisions:
        f, day = d["family"], _ts(d["decided_at"]).date()
        x = fam[f]
        x["n"] += 1
        before = d["outcome"] == "funded"
        after = before
        w_off = off.get(day, {}).get(f)
        w_on = on.get(day, {}).get(f)
        rank, charge, w_led = d.get("value_rank"), d.get("charge"), d.get("family_weight")
        if d["reason"] in WEIGHT_DEPENDENT and None not in (rank, charge, w_led) and w_on is not None:
            x["weight_dependent"] += 1
            x["w_ledger"].append(float(w_led))
            x["w_off"].append(w_off.weight)
            x["w_on"].append(w_on.weight)
            repro_n += 1
            repro_ok += abs(w_off.weight - float(w_led)) <= REPRO_TOL
            w_new = float(w_led) + (w_on.weight - w_off.weight)
            x["w_new"].append(w_new)
            if d["reason"] != "exploration_floor":
                after = w_new * float(rank) - float(charge) > 0
        x["funded_before"] += before
        x["funded_after"] += after
        x["gained"] += (after and not before)
        x["lost"] += (before and not after)
    tier_of = {f: families.get(f).tier if families.get(f) else None for f in fam}
    t2_before = sum(v["funded_before"] for f, v in fam.items() if tier_of[f] == 2)
    t2_after = sum(v["funded_after"] for f, v in fam.items() if tier_of[f] == 2)
    out: dict[str, dict] = {}
    starved = []
    last_on = on[end]
    for f, v in sorted(fam.items()):
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        ev = last_on[f].evidence if f in last_on else None
        row = {k: v[k] for k in ("n", "weight_dependent", "funded_before", "funded_after", "gained", "lost")}
        row.update(tier=tier_of[f], w_ledger=mean(v["w_ledger"]), w_off=mean(v["w_off"]), w_on=mean(v["w_on"]), w_counterfactual=mean(v["w_new"]),
                   reliability_rate=round(ev.reliability_rate, 4) if ev and ev.reliability_rate is not None else None,
                   reliability_n=ev.reliability_n if ev else 0,
                   basis_on=last_on[f].basis if f in last_on else None)
        if tier_of[f] == 2 and t2_before and t2_after:
            row["t2_share_before"] = round(v["funded_before"] / t2_before, 4)
            row["t2_share_after"] = round(v["funded_after"] / t2_after, 4)
            if v["funded_before"] and row["t2_share_after"] < FLOOR_SHARE <= row["t2_share_before"]:
                starved.append(f)
        out[f] = row
    return {"decisions": len(decisions), "days": [start.isoformat(), end.isoformat()], "families": out,
            "repro_share": round(repro_ok / repro_n, 4) if repro_n else None, "repro_n": repro_n,
            "starved": starved,
            "max_daily_move_on": round(max_daily_move(weight_series(families, acts, uses, start=start, end=end,
                                                                    reliability=True)), 6),
            "priced_weight_dependent": sum(1 for d in decisions if d["reason"] in WEIGHT_DEPENDENT
                                           and (d.get("charge") or 0) > 0),
            "event_wants_in_window": sum(1 for d in decisions if d.get("salience") is not None)}


# --- fixture lane --------------------------------------------------------------------------------

def _fixture() -> int:
    fams = parse_families({"version": 1, "families": {
        "steady": {"tier": 2, "deferrable_window_s": 3600},
        "flaky": {"tier": 2, "deferrable_window_s": 3600},
        "quiet": {"tier": 2, "deferrable_window_s": 3600},
    }})
    now = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    acts: list[ActRecord] = []
    for i in range(80):
        t = now - timedelta(days=20) + timedelta(hours=6 * i)
        acts.append(ActRecord(act_id=f"s{i}", family="steady", opened_at=t - timedelta(minutes=5), closed_at=t,
                              outcome_value=1.0, outcome_source="producer"))
        acts.append(ActRecord(act_id=f"f{i}", family="flaky", opened_at=t - timedelta(minutes=5), closed_at=t,
                              outcome_value=0.0 if i % 5 else 1.0, outcome_source="producer"))
    decisions = []
    for k in range(14):
        t = now - timedelta(days=13 - k)
        for f in ("steady", "flaky", "quiet"):
            decisions.append({"decided_at": t, "family": f, "tier": 2, "outcome": "funded", "reason": "net_positive",
                              "family_weight": 1.0, "value_rank": 0.5, "charge": 0.45})
    r = replay(fams, decisions, acts, [])
    fs = r["families"]
    checks = {
        "off_reproduces_recorded_weight": r["repro_share"] == 1.0,
        "reliable_family_gains_weight": fs["steady"]["w_on"] > 1.0,
        "unreliable_family_loses_weight_and_funding": fs["flaky"]["w_on"] < 1.0 and fs["flaky"]["lost"] > 0,
        "within_drift_cap": r["max_daily_move_on"] <= MAX_DRIFT_PER_DAY + 1e-9,
        "no_evidence_equals_off": fs["quiet"]["reliability_n"] == 0,
    }
    print(json.dumps({"lane": "fixture", "checks": checks, "families": fs}, indent=2, default=str))
    return 0 if all(checks.values()) else 1


# --- live lane -----------------------------------------------------------------------------------

def _live(days: int, out_json: Optional[str]) -> int:
    import _body_economy_data as data_mod

    q = data_mod.run_query
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    hs = start - EVIDENCE_WINDOW
    iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    decisions = q(f"""select coalesce(json_agg(r order by r.decided_at), '[]'::json) from (
        select decided_at, family, tier, outcome, reason,
               (decision->>'family_weight')::float as family_weight, (decision->>'value_rank')::float as value_rank,
               (decision->>'charge')::float as charge, (decision->'want'->>'salience')::float as salience
          from want_ledger where decided_at >= '{iso(start)}') r;""")
    acts_rows = q(f"""select coalesce(json_agg(r), '[]'::json) from (
        select act_id, actor, opened_at, closed_at, outcome_value, outcome_source, outcome_note, signal_id
          from grammar_acts where coalesce(opened_at, closed_at, created_at) >= '{iso(hs)}') r;""")
    use_rows = q(f"""select coalesce(json_agg(r), '[]'::json) from (
        select act_id, actor, started_at, link_status, used, window_closed
          from act_downstream_use where started_at >= '{iso(hs)}') r;""")
    acts = [ActRecord(act_id=str(r["act_id"]), family=str(r["actor"]), opened_at=_ts(r.get("opened_at")),
                      closed_at=_ts(r.get("closed_at")),
                      outcome_value=float(r["outcome_value"]) if r.get("outcome_value") is not None else None,
                      outcome_source=r.get("outcome_source"), signal_id=r.get("signal_id"),
                      outcome_note=r.get("outcome_note")) for r in acts_rows]
    uses = [UseRecord(act_id=str(r["act_id"]), family=str(r["actor"]), started_at=_ts(r.get("started_at")),
                      link_status=str(r["link_status"]), used=r.get("used"), window_closed=bool(r.get("window_closed")))
            for r in use_rows]
    families, _ = load_families(ROOT / "config" / "physiology" / "families.yaml")
    r = replay(families, decisions, acts, uses)
    checks = {
        "no_tier2_family_starved": not r["starved"],
        "within_drift_cap": r["max_daily_move_on"] <= MAX_DRIFT_PER_DAY + 1e-9,
    }
    report = {"lane": "live", "window_days": days, "checks": checks, **r}
    text = json.dumps(report, indent=2, default=str)
    if out_json:
        Path(out_json).write_text(text)
    print(text)
    return 0 if all(checks.values()) else 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out-json")
    a = ap.parse_args(argv)
    return _live(a.days, a.out_json) if a.live else _fixture()


if __name__ == "__main__":
    raise SystemExit(main())
