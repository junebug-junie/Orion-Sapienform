#!/usr/bin/env python3
"""Attend -> act -> learn loop eval (attend-to-act design D4 + "Amendment 2026-09-29").

Fixture mode (CI, default): a complete treated chain and a complete control chain must pass
``check_chain``; every rule must fire on its own mutation (missing link, Orion writing ``resolved``,
a winner older than 90 s at bind, a shed overlapping an open cooling incident without
``preempted_by_reflex``, a proposal while a hardware-watch incident was open, an ``overlap:reflex``
row reaching the posterior). Plus the pure acceptance step check 1 (cabinet_heat_pressure scale).
Check 3 (the dispatch allocator's information floor) retired with the allocator on 2026-11-07
(ladder retirement PR 6): the want market decides the act now.

Live mode (``--live``, needs ORION_PG_DSN): reads the last ``--days`` (7) of real episodes, runs
``check_chain`` on each market episode and reports link completeness, the treated-vs-control
contrast three ways, check 1 against real cabinet rows and the warming signal's live sanity.
Old dispatch-path episodes (before 2026-11-05) are listed but not re-checked: their proposal,
policy and dispatch tables are retired (2026-11-07). Read-only transaction, statement timeout.
Episodes that do not exist yet are reported as such -- never as a pass.

    python orion/autonomy/evals/run_attend_act_loop_eval.py
    ORION_PG_DSN=... python orion/autonomy/evals/run_attend_act_loop_eval.py --live
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orion.autonomy.attend_act_loop import check_chain, contrast, is_market_episode  # noqa: E402
from orion.autonomy.cabinet_heat import cabinet_heat_pressure, read_cabinet_heat  # noqa: E402
from orion.hardware_watch.rules import TempPoint  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures/attend_act_loop_chain.json"


def fixture_checks() -> list[tuple[str, bool]]:
    src = json.loads(FIXTURE.read_text())
    treated, control = src["treated"], src["control"]
    checks = [("complete_treated_chain_passes", check_chain(treated) == []),
              ("complete_control_chain_passes", check_chain(control) == [])]

    def fails(name, chain, expect):
        got = check_chain(chain)
        checks.append((name, any(v.startswith(expect) for v in got)))

    for link in ("broadcast_row", "proposal", "decision", "dispatch", "pool_shed", "result", "outcome_row"):
        m = deepcopy(treated)
        m[link] = None
        fails(f"missing_{link}_fails", m, "missing_link")
    m = deepcopy(treated)
    m["loop_outcomes"] = []
    fails("missing_loop_outcome_fails", m, "missing_link:loop_outcome")
    m = deepcopy(treated)
    m["loop_outcomes"].append({"verdict": "resolved", "actor": "orion"})
    fails("orion_resolved_fails", m, "orion_wrote_terminal_verdict")
    m = deepcopy(treated)
    m["broadcast_row"]["generated_at"] = "2026-10-01T11:58:00+00:00"   # 120 s before decided_at
    fails("winner_older_than_90s_fails", m, "winner_stale_at_bind")
    m = deepcopy(treated)
    m["incidents"] = [{"rule": "cooling", "opened_at": "2026-10-01T12:05:00+00:00"}]
    fails("reflex_overlap_not_preempted_fails", m, "shed_overlapped_reflex_not_preempted")
    m = deepcopy(treated)
    m["episode"]["eligibility"]["hardware_watch"]["open_incident_ids"] = ["abc123abc123"]
    fails("proposal_while_incident_open_fails", m, "proposed_while_hardware_incident_open")
    m = deepcopy(control)
    m["episode"]["outcome"]["overlap"] = ["overlap:reflex"]
    fails("reflex_overlap_reaching_posterior_fails", m, "reflex_overlap_reached_posterior")
    m = deepcopy(control)
    m["loop_outcomes"] = [{"verdict": "acted", "actor": "orion"}]
    fails("acted_on_control_arm_fails", m, "acted_verdict_on_control_arm")
    m = deepcopy(treated)
    m["episode"]["created_at"] = "2026-10-01T12:00:05+00:00"
    fails("late_expectation_fails", m, "expectation_not_precommitted")
    # A preempted shed is a legitimate outcome: recorded, excluded, still a closed chain.
    m = deepcopy(treated)
    m["incidents"] = [{"rule": "cooling", "opened_at": "2026-10-01T12:05:00+00:00"}]
    m["pool_shed"]["state"] = "preempted_by_reflex"
    m["episode"]["settlement_state"] = "preempted_by_reflex"
    m["episode"]["outcome"] = {"excluded_reason": "overlap:reflex", "overlap": ["overlap:reflex"], "posterior_updated": False}
    m["outcome_row"] = None
    checks.append(("preempted_by_reflex_chain_is_clean", check_chain(m) == []))

    # The market chain (ladder retirement PR 5): the cooling act decided by the want market.
    mt, mc = src["market_treated"], src["market_control"]
    checks += [("complete_market_treated_chain_passes", check_chain(mt) == []),
               ("complete_market_control_chain_passes", check_chain(mc) == [])]
    for link in ("want_decision", "pool_shed", "outcome_row"):
        m = deepcopy(mt)
        m[link] = None
        fails(f"market_missing_{link}_fails", m, "missing_link")
    m = deepcopy(mc)
    m["want_decision"] = {"outcome": "deferred", "reason": "sated"}
    fails("market_episode_from_a_deferral_fails", m, "episode_without_a_funded_decision")
    m = deepcopy(mt)
    m["episode"]["created_at"] = "2026-10-02T12:00:05+00:00"
    fails("market_late_expectation_fails", m, "expectation_not_precommitted")

    # Check 1: the outcome scale, on the thermal gate's own constants.
    checks.append(("check1_pressure_scale", cabinet_heat_pressure(27.9) == 0.0 and cabinet_heat_pressure(29.5) == 0.375
                   and cabinet_heat_pressure(32.0) == 1.0))
    return checks


def live(days: float) -> dict:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    dsn = os.environ.get("ORION_PG_DSN") or os.environ.get("POSTGRES_URI")
    if not dsn:
        raise SystemExit("ORION_PG_DSN required for --live")
    since = datetime.now(timezone.utc) - timedelta(days=days)
    report: dict = {"mode": "live", "days": days}
    with psycopg2.connect(dsn, options="-c default_transaction_read_only=on -c statement_timeout=60000") as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT timestamp::timestamptz AS ts, (measurements->>'cabinet_temp_c')::float AS v "
                    "FROM orion_biometrics_summary WHERE node='athena' AND timestamp >= %s AND measurements ? 'cabinet_temp_c' "
                    "ORDER BY timestamp", (since.strftime("%Y-%m-%d %H:%M:%S"),))
        points = [TempPoint(r["ts"], r["v"]) for r in cur.fetchall() if r["v"] is not None]
        pressures = [cabinet_heat_pressure(p.value) for p in points]
        report["check1_live"] = {
            "samples": len(points),
            "share_at_zero": round(sum(1 for x in pressures if x == 0.0) / len(pressures), 4) if pressures else None,
            "share_saturated": round(sum(1 for x in pressures if x == 1.0) / len(pressures), 4) if pressures else None,
            "zero_below_28": all(cabinet_heat_pressure(p.value) == 0.0 for p in points if p.value < 28.0),
        }
        # The warming (attention) signal on a 1-minute grid: how often non-zero, does it return to 0.
        if points:
            ts = [p.ts for p in points]
            import bisect
            grid, t = [], points[0].ts + timedelta(hours=2)
            while t <= points[-1].ts:
                lo, hi = bisect.bisect_left(ts, t - timedelta(hours=2)), bisect.bisect_right(ts, t)
                grid.append(read_cabinet_heat(points[lo:hi], t).warming_error)
                t += timedelta(minutes=1)
            episodes = sum(1 for i, v in enumerate(grid) if v > 0 and (i == 0 or grid[i - 1] == 0))
            report["warming_signal_live"] = {"minutes": len(grid), "nonzero_minutes": sum(1 for v in grid if v > 0),
                                             "episodes": episodes, "returns_to_zero": bool(grid) and grid[-1] == 0.0 or 0.0 in grid}
        cur.execute("SELECT to_regclass('substrate_world_action_episodes') AS t")
        if cur.fetchone()["t"] is None:
            report["episodes"] = "none_yet (ledger table absent -- no world decision has been made)"
            chains = []
        else:
            cur.execute("SELECT * FROM substrate_world_action_episodes WHERE decided_at >= %s ORDER BY decided_at", (since,))
            eps = [dict(r) for r in cur.fetchall()]
            chains = []
            for ep in eps:
                ch: dict = {"episode": ep}
                if is_market_episode(ep):
                    # The cooling act since 2026-11-05: the market's decision, the pool's shed, the outcome row.
                    cur.execute("SELECT outcome, reason, decision_id FROM want_ledger WHERE decision_id = %s LIMIT 1",
                                (ep.get("decision_id"),))
                    ch["want_decision"] = cur.fetchone()
                    ch["pool_shed"] = None
                    if ep["arm"] == "treated":
                        cur.execute("SELECT * FROM gpu_pool_orion_shed WHERE dispatch_id = %s", (ep["episode_id"],))
                        ch["pool_shed"] = cur.fetchone()
                    cur.execute("SELECT id FROM substrate_action_outcomes WHERE dispatch_id = %s "
                                "AND signal_id = 'cabinet_heat_pressure'", (ep["episode_id"],))
                    ch["outcome_row"] = cur.fetchone()
                    cur.execute("SELECT rule, opened_at FROM hardware_watch_incident WHERE opened_at >= %s AND opened_at <= %s",
                                (ep["decided_at"], ep["decided_at"] + timedelta(minutes=25)))
                    ch["incidents"] = [dict(r) for r in cur.fetchall()]
                    chains.append(ch)
                    continue
                # A dispatch-path episode (before 2026-11-05): its proposal, policy and dispatch hops lived in
                # tables retired on 2026-11-07, so its chain is not re-read. Listed, never counted as closed.
                ch["ladder_path"] = True
                chains.append(ch)
            report["episodes"] = [{"episode_id": c["episode"]["episode_id"], "arm": c["episode"]["arm"],
                                   "settlement_state": c["episode"].get("settlement_state"),
                                   "scored": c["episode"].get("scored_at") is not None,
                                   "violations": (["ladder_path_not_rechecked"] if c.get("ladder_path")
                                                  else check_chain(c) if c["episode"].get("scored_at") else ["not_scored_yet"])}
                                  for c in chains]
            report["closed_chains"] = sum(1 for e in report["episodes"] if not e["violations"])
            report["contrast"] = contrast([c["episode"] for c in chains if c["episode"].get("scored_at")])
    report["verdict"] = ("UNVERIFIED: no closed chain on the live rail yet" if not report.get("closed_chains")
                         else f"{report['closed_chains']} closed chain(s); effect UNVERIFIED below D3 volume")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="attend-to-act loop eval")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=7.0)
    args = ap.parse_args()
    if args.live:
        print(json.dumps(live(args.days), indent=2, default=str))
        return 0
    checks = fixture_checks()
    print(json.dumps({"eval": "attend_act_loop_chain", "checks": dict(checks),
                      "passed": sum(ok for _, ok in checks), "total": len(checks)}, indent=2))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
