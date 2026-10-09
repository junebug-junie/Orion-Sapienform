#!/usr/bin/env python3
"""Attend -> act -> learn loop eval (attend-to-act design D4 + "Amendment 2026-09-29").

Fixture mode (CI, default): a complete treated chain and a complete control chain must pass
``check_chain``; every rule must fire on its own mutation (missing link, Orion writing ``resolved``,
a winner older than 90 s at bind, a shed overlapping an open cooling incident without
``preempted_by_reflex``, a proposal while a hardware-watch incident was open, an ``overlap:reflex``
row reaching the posterior). Plus the pure acceptance steps: check 1 (cabinet_heat_pressure scale)
and check 3 (the action clears the allocator floor honestly, cold and warm, and what charging the
TTL as motor cost would do).

Live mode (``--live``, needs ORION_PG_DSN): reads the last ``--days`` (7) of real episodes, runs
``check_chain`` on each and reports link completeness, the treated-vs-control contrast three ways,
check 1 against real cabinet rows, the warming signal's live sanity, and check 8's "zero proposals
made while a hardware-watch incident was open". Read-only transaction, statement timeout. Episodes
that do not exist yet are reported as such -- never as a pass.

    python orion/autonomy/evals/run_attend_act_loop_eval.py
    ORION_PG_DSN=... python orion/autonomy/evals/run_attend_act_loop_eval.py --live
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orion.autonomy.allocator import allocate, candidate_from_dispatch, expected_information_gain_nats  # noqa: E402
from orion.autonomy.attend_act_loop import check_chain, contrast  # noqa: E402
from orion.autonomy.cabinet_heat import cabinet_heat_pressure, read_cabinet_heat  # noqa: E402
from orion.autonomy.prediction import DEFAULT_OBSERVATION_VARIANCE, DEFAULT_PRIOR_VARIANCE  # noqa: E402
from orion.hardware_watch.rules import TempPoint  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures/attend_act_loop_chain.json"
MIN_NATS_PER_SEC = 0.02   # ORION_DISPATCH_MIN_NATS_PER_SEC, unchanged by this design


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

    # Check 1: the outcome scale, on the thermal gate's own constants.
    checks.append(("check1_pressure_scale", cabinet_heat_pressure(27.9) == 0.0 and cabinet_heat_pressure(29.5) == 0.375
                   and cabinet_heat_pressure(32.0) == 1.0))
    # Check 3: the action clears the UNCHANGED floor honestly (executor wall time = one RPC).
    nats_cold = expected_information_gain_nats(DEFAULT_PRIOR_VARIANCE, DEFAULT_OBSERVATION_VARIANCE)
    ok = True
    for cost in (1.0, 2.0, 5.0):   # 5.0 = ORION_DISPATCH_MOTOR_TYPICAL_COST_SEC before any cost sample exists
        c = candidate_from_dispatch(dispatch_id=f"d{cost}", dispatch_kind="self_regulate", target_id="pool:background_gpu",
                                    signal_id="cabinet_heat_pressure", claimed_direction="decrease",
                                    cell_variances_by_volume=[], cost_sec=cost, cold_variance=DEFAULT_PRIOR_VARIANCE)
        alloc = allocate([c], allowance_sec=3600.0, min_nats_per_sec=MIN_NATS_PER_SEC)
        ok = ok and [a.dispatch_id for a in alloc.admitted] == [c.dispatch_id]
    # Warm (not cold-start): after n treated observations the pooled variance shrinks; at 2 s it must
    # still clear the floor for the first ~12 settled rows (design: retires after ~12-24).
    var12 = 1.0 / (1.0 / DEFAULT_PRIOR_VARIANCE + 12 / DEFAULT_OBSERVATION_VARIANCE)
    warm = candidate_from_dispatch(dispatch_id="warm", dispatch_kind="self_regulate", target_id="pool:background_gpu",
                                   signal_id="cabinet_heat_pressure", claimed_direction="decrease",
                                   cell_variances_by_volume=[(var12, 12)], cost_sec=1.0, cold_variance=DEFAULT_PRIOR_VARIANCE)
    warm_alloc = allocate([warm], allowance_sec=3600.0, min_nats_per_sec=MIN_NATS_PER_SEC)
    ttl_charged = nats_cold / 900.0
    checks.append(("check3_clears_floor_cold_and_warm", ok and bool(warm_alloc.admitted)
                   and nats_cold / 2.0 >= 0.4 and ttl_charged < MIN_NATS_PER_SEC))
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
                cur.execute("SELECT generated_at, projection_json->>'selected_open_loop_id' AS selected_open_loop_id, "
                            "(projection_json->>'dwell_ticks')::int AS dwell_ticks FROM substrate_attention_broadcast_log "
                            "WHERE log_id = %s", (ep.get("broadcast_log_id"),))
                ch["broadcast_row"] = cur.fetchone()
                cur.execute("SELECT 1 FROM substrate_proposal_frames f, jsonb_array_elements(f.proposal_frame_json->'candidates') c "
                            "WHERE c->>'proposal_id' = %s LIMIT 1", (ep.get("proposal_id"),))
                ch["proposal"] = cur.fetchone()
                cur.execute("SELECT 1 FROM substrate_policy_decision_frames f, jsonb_array_elements(f.policy_decision_frame_json->'decisions') d "
                            "WHERE d->>'decision_id' = %s LIMIT 1", (ep.get("decision_id"),))
                ch["decision"] = cur.fetchone()
                cur.execute("SELECT c->>'dispatched_at' AS dispatched_at FROM substrate_execution_dispatch_frames f, "
                            "jsonb_array_elements(f.dispatch_frame_json->'dispatched_candidates' || f.dispatch_frame_json->'blocked_candidates') c "
                            "WHERE f.frame_id = %s AND c->>'dispatch_id' = %s LIMIT 1", (ep.get("dispatch_frame_id"), ep["episode_id"]))
                ch["dispatch"] = cur.fetchone()
                ch["pool_shed"] = None
                if ep["arm"] == "treated":
                    cur.execute("SELECT * FROM gpu_pool_orion_shed WHERE dispatch_id = %s", (ep["episode_id"],))
                    ch["pool_shed"] = cur.fetchone()
                cur.execute("SELECT result_id FROM substrate_dispatch_results WHERE dispatch_id = %s", (ep["episode_id"],))
                ch["result"] = cur.fetchone()
                cur.execute("SELECT id FROM substrate_action_outcomes WHERE dispatch_id = %s AND signal_id = 'cabinet_heat_pressure'",
                            (ep["episode_id"],))
                ch["outcome_row"] = cur.fetchone()
                cur.execute("SELECT verdict, actor FROM attention_loop_outcome WHERE loop_id = %s AND created_at >= %s",
                            (ep.get("open_loop_id"), ep["decided_at"]))
                ch["loop_outcomes"] = [dict(r) for r in cur.fetchall()]
                cur.execute("SELECT rule, opened_at FROM hardware_watch_incident WHERE opened_at >= %s AND opened_at <= %s",
                            (ep["decided_at"], ep["decided_at"] + timedelta(minutes=25)))
                ch["incidents"] = [dict(r) for r in cur.fetchall()]
                chains.append(ch)
            report["episodes"] = [{"episode_id": c["episode"]["episode_id"], "arm": c["episode"]["arm"],
                                   "settlement_state": c["episode"].get("settlement_state"),
                                   "scored": c["episode"].get("scored_at") is not None,
                                   "violations": check_chain(c) if c["episode"].get("scored_at") else ["not_scored_yet"]}
                                  for c in chains]
            report["closed_chains"] = sum(1 for e in report["episodes"] if not e["violations"])
            report["contrast"] = contrast([c["episode"] for c in chains if c["episode"].get("scored_at")])
        cur.execute("SELECT count(*) AS n FROM substrate_proposal_frames f, jsonb_array_elements(f.proposal_frame_json->'candidates') c "
                    "WHERE f.created_at >= %s AND c->'execution_intent'->>'template' = 'shed_background_gpu' "
                    "AND jsonb_array_length(COALESCE(c->'world_eligibility'->'hardware_watch'->'open_incident_ids', '[]'::jsonb)) > 0",
                    (since,))
        report["check8_proposals_while_incident_open"] = cur.fetchone()["n"]
        cur.execute("SELECT count(*) AS n FROM substrate_proposal_frames f, jsonb_array_elements(f.proposal_frame_json->'candidates') c "
                    "WHERE f.created_at >= %s AND c->'execution_intent'->>'template' = 'shed_background_gpu'", (since,))
        report["shed_proposals"] = cur.fetchone()["n"]
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
