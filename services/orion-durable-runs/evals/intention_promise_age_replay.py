#!/usr/bin/env python3
"""Eval: does a take-on's carry run bid at the AGE OF THE PROMISE? (#3640, Juniper 2026-11-18). READ-ONLY.

The market rank of a ``durable.intention`` want is ``1 - exp(-age / tau)``, age = now - the promise's time
(``brief.promised_at``), tau = the family's 86400 s window (``hunger_from: want_opened`` in
config/physiology/families.yaml). Before: age = now - the run's own request, so a bid 5 s after Prime's take-on
ranked ~0 and the pool's value-ordered background queue put it behind every positive-net agent run.

Two lanes:

* fixture (default; deterministic, no database): the 11-13 run as stored (durable_admission_runs request +
  harness_turn_trace promise time) replayed through the shipped ``want_for_run`` -> ``hunger`` -> ``rank_value``;
  a promise-age ladder (monotone, ~0 when fresh, never reset by another promise's close); and the replayed net
  against a 7-day snapshot of other agent-lane families' median nets (taken 2026-11-18): the old net sits below all
  of them, the new one above the medians.
* ``--live --dsn <postgres dsn>``: every ``intention.carry`` run in ``durable_admission_runs``: its first
  ``want_ledger`` decision (what the market said), the promise's time from its brief's ``quote_ref`` (the
  ``harness_turn_trace`` row's ``created_at``), and the rank the shipped rule gives at the same decision time.

    PYTHONPATH=. .venv/bin/python services/orion-durable-runs/evals/intention_promise_age_replay.py [--live --dsn ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC)]

from app.run_bids import want_for_run  # noqa: E402
from orion.physiology.appetite import ActRecord, build_family_appetite, hunger, rank_value  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.schemas.intention import INTENTION_FAMILY  # noqa: E402

FAMS, _ = load_families(ROOT / "config/physiology/families.yaml")
CFG = FAMS.get(INTENTION_FAMILY)
LIVE_WEIGHT = 1.03                       # the 11-13 decision's weight (prior)
# want_ledger, 7 days to 2026-11-18, median net per agent-lane durable family (n >= 29).
AGENT_MEDIAN_NETS = {"durable.pr_field_note": 0.352, "durable.self_sense_eval": 0.516,
                     "durable.curiosity_investigate": 0.720, "durable.self_inquiry": 0.542,
                     "durable.session_roundup": 0.538}


def _ts(s) -> datetime:
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def market_rank(request: dict, at: datetime, acts=()) -> dict:
    want = want_for_run(request, CFG.tier)
    app = build_family_appetite(CFG, acts, at)
    h = hunger(app, at, opened_at=want.opened_at)
    rank, basis, _ = rank_value(app, h.intensity, want.outcome_value)
    return {"opened_at": want.opened_at.isoformat(), "age_sec": round(h.unfunded_sec or 0.0, 1),
            "hunger_basis": h.basis, "rank": round(rank, 6), "rank_basis": basis}


def _request(*, run_id, requested_at, promised_at, horizon) -> dict:
    brief = {"intention": "intention.carry", "concern_id": run_id.split("-", 1)[1], "source": "turn_promise",
             "prime_act_id": "act", "prompt": "p", "route": "chat_thread", "horizon_at": horizon}
    if promised_at is not None:
        brief["promised_at"] = promised_at
    return {"run_id": run_id, "workflow": "intention.carry", "requested_at": requested_at, "brief": brief,
            "admission": {"deadline_at": horizon}}


def fixture() -> tuple[dict, list[str]]:
    fails: list[str] = []
    out: dict = {}
    # The 11-13 run (#3640): durable_admission_runs request, harness_turn_trace a129b72f-... created_at.
    decided = _ts("2026-11-13T19:47:13.669909Z")
    req = _request(run_id="intention-d8db9d7b1bba4cf7", requested_at="2026-11-13T19:47:08.522865Z",
                   promised_at="2026-11-12T07:08:40.440369Z", horizon="2026-11-15T07:08:40.440369Z")
    new = market_rank(req, decided)
    old = market_rank({**req, "brief": {k: v for k, v in req["brief"].items() if k != "promised_at"}}, decided)
    out["replay_11_13"] = {"live_decision": {"rank": 0.0, "net": 0.000062, "hunger": "5 s since_opened"},
                           "without_promise_time": old, "shipped": new,
                           "net_shipped_charge_0": round(LIVE_WEIGHT * new["rank"], 4)}
    if not old["rank"] < 1e-3:
        fails.append(f"replay: without the promise time the rank should be ~0 (live 0.00), got {old['rank']}")
    if not abs(new["rank"] - 0.783) < 0.002 or new["hunger_basis"] != "since_opened":
        fails.append(f"replay: shipped rank should be 0.783 from a 36.6 h old promise, got {new}")
    beaten_old = sorted(f for f, m in AGENT_MEDIAN_NETS.items() if LIVE_WEIGHT * old["rank"] < m)
    beaten_new = sorted(f for f, m in AGENT_MEDIAN_NETS.items() if LIVE_WEIGHT * new["rank"] < m)
    out["agent_families_with_a_higher_median_net"] = {"before": beaten_old, "shipped": beaten_new}
    if len(beaten_old) != len(AGENT_MEDIAN_NETS) or beaten_new:
        fails.append(f"queue: before should sit under every agent median, shipped under none: {beaten_old} / {beaten_new}")

    # Ladder: a promise made N hours before a bid 5 s after the order; another carry closed 10 min before the bid.
    bid = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
    closed = [ActRecord(act_id="a0", family=INTENTION_FAMILY, opened_at=bid - timedelta(hours=2),
                        closed_at=bid - timedelta(minutes=10))]
    ladder = []
    for hours in (0.0167, 1, 6, 12, 24, 48, 71):
        r = market_rank(_request(run_id="intention-0123456789abcdef", requested_at=(bid - timedelta(seconds=5)).isoformat(),
                                 promised_at=(bid - timedelta(hours=hours)).isoformat(),
                                 horizon=(bid + timedelta(hours=72 - hours)).isoformat()), bid, closed)
        ladder.append({"promise_age_h": hours, "rank": r["rank"]})
        want = 1 - math.exp(-hours * 3600 / CFG.deferrable_window_s)
        if abs(r["rank"] - want) > 1e-6:
            fails.append(f"ladder: {hours} h -> {r['rank']}, want {want:.6f} (a close must not reset it)")
    out["ladder"] = ladder
    if ladder[0]["rank"] > 1e-3:
        fails.append(f"ladder: a 1-minute-old promise must still read ~0, got {ladder[0]['rank']}")
    return out, fails


LIVE_SQL = """
SELECT r.run_id, r.request,
       (SELECT w.decided_at FROM want_ledger w WHERE w.want_id = r.run_id ORDER BY w.decided_at LIMIT 1) AS decided_at,
       (SELECT w.decision->>'value_rank' FROM want_ledger w WHERE w.want_id = r.run_id ORDER BY w.decided_at LIMIT 1) AS live_rank,
       (SELECT w.net FROM want_ledger w WHERE w.want_id = r.run_id ORDER BY w.decided_at LIMIT 1) AS live_net,
       (SELECT t.created_at FROM harness_turn_trace t
         WHERE t.correlation_id = split_part(split_part(r.request->'brief'->>'quote_ref', ':', 2), '#', 1)) AS promised_at
FROM durable_admission_runs r
WHERE r.request->>'workflow' = 'intention.carry'
ORDER BY r.created_at
"""


def live(dsn: str) -> tuple[list[dict], list[str]]:
    import psycopg
    from psycopg.rows import dict_row

    rows, fails = [], []
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        for r in conn.execute(LIVE_SQL).fetchall():
            req = r["request"] if isinstance(r["request"], dict) else json.loads(r["request"])
            at = _ts(r["decided_at"]) if r["decided_at"] else _ts(req["requested_at"])
            if r["promised_at"] is not None and not req["brief"].get("promised_at"):
                req = {**req, "brief": {**req["brief"], "promised_at": _ts(r["promised_at"]).isoformat()}}
            shipped = market_rank(req, at)
            rows.append({"run_id": r["run_id"], "decided_at": at.isoformat(),
                         "promised_at": _ts(r["promised_at"]).isoformat() if r["promised_at"] else None,
                         "live_rank": r["live_rank"], "live_net": r["live_net"], "shipped": shipped})
            if r["promised_at"] is None:
                fails.append(f"{r['run_id']}: no harness_turn_trace row for its quote_ref (promise time unknown)")
    return rows, fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--dsn", default=None, help="read-only Postgres DSN (required with --live)")
    a = ap.parse_args()
    report, fails = fixture()
    out = {"fixture": report}
    if a.live:
        if not a.dsn:
            ap.error("--live needs --dsn")
        out["live"], live_fails = live(a.dsn)
        fails += live_fails
    out["fails"], out["ok"] = fails, not fails
    print(json.dumps(out, indent=1, default=str))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
