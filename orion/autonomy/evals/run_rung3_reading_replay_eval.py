#!/usr/bin/env python3
"""Thinking rung 3 follow-up replay eval: the reading pull and the holdback rule.

Fixture mode (CI, default): deterministic checks that the shipped numbers are the derived ones
and the rules decide what the design says:
  * the information floor is still 0.02 nats/s;
  * the daily cap default (Hub settings, .env_example, pull.DERIVED_DAILY_CAP) equals
    derive_daily_cap over the recorded live series; the route horizon is 12000 s;
  * a painting decision is never held back, a reading pull is (holdback_rule);
  * the live look act (n=119, mean 0.696) at its live cost (679 s) is refused at the floor and
    admitted on the proven path.

Live mode (``--live``, needs ORION_PG_DSN; read-only, statement timeout; reads Redis
orion:regulation:latest from ORION_BUS_URL when reachable). For the last ``--hours`` (48):
  1. re-derive the daily cap (30 days of Stage 1 reads per UTC day) and the horizon (p90 of Stage 1
     submission -> look close);
  2. replay one reading-pull candidate per proposal tick (the ticks render_scene, the same
     no-dimension template shape, reached the dispatch runtime) through both admission paths with
     the time-correct look posterior and cost, the in-flight rule (12000 s, either arm), Hub's
     gates in order, the hold on refusal and the daily cap; report admissions per day per path,
     holdback draws, cap hits and check RPCs, as lived and with arousal counterfactually calm.

    python orion/autonomy/evals/run_rung3_reading_replay_eval.py
    ORION_PG_DSN=... ORION_BUS_URL=redis://100.92.216.81:6379/0 \
        python orion/autonomy/evals/run_rung3_reading_replay_eval.py --live --hours 48
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orion.autonomy import act_episodes  # noqa: E402
from orion.autonomy.act_candidates import ActRef  # noqa: E402
from orion.autonomy.evals.run_rung3_replay_eval import (  # noqa: E402
    _aware,
    decide,
    rebuild_posterior,
    settings_floor,
    stable_draw,
)
from orion.autonomy.prediction import EffectPosterior  # noqa: E402
from orion.execution_dispatch.policy import READ_PULL_EXECUTOR_VERB, load_execution_dispatch_policy  # noqa: E402
from orion.world_pulse_read import pull  # noqa: E402

HUB_SETTINGS = ROOT / "services/orion-hub/app/settings.py"
HUB_ENV = ROOT / "services/orion-hub/.env_example"
POLICY = ROOT / "config/execution_dispatch/execution_dispatch_policy.v1.yaml"
RENDER_VERB = "skills.imagination.render_scene.v1"
LOOK_KEY = ("act:look", "hub.world_pulse_read", "artifact_produced")
COST_LOOKBACK = timedelta(days=30)
FRACTION = 0.2
# Recorded 2026-10-21 (DAILY_READS_STARTED_SQL over 30 days): the series the default rests on.
RECORDED_DAILY_READS = [1, 12, 15, 27, 5, 6, 6, 8, 8, 5, 5, 6, 2, 2, 23, 6, 4, 8, 46, 49, 45, 22, 13, 27, 15]


def _read_ref() -> ActRef:
    a = load_execution_dispatch_policy(POLICY).act_refs_by_cortex_verb()[READ_PULL_EXECUTOR_VERB]
    return ActRef(verb=a.verb, actor=a.actor, signal_id=a.signal_id, horizon_sec=a.horizon_sec,
                  object_source=a.object_source)


def shipped_cap_defaults() -> tuple[int, int]:
    s = re.search(r"HUB_WORLD_PULSE_READ_PULL_DAILY_CAP: int = Field\(\s*default=(\d+)", HUB_SETTINGS.read_text())
    e = re.search(r"^HUB_WORLD_PULSE_READ_PULL_DAILY_CAP=(\d+)$", HUB_ENV.read_text(), re.M)
    if not s or not e:
        raise SystemExit("could not read HUB_WORLD_PULSE_READ_PULL_DAILY_CAP defaults")
    return int(s.group(1)), int(e.group(1))


def fixture_checks() -> list[tuple[str, bool]]:
    floor = settings_floor()
    ref = _read_ref()
    settings_cap, env_cap = shipped_cap_defaults()
    derived = pull.derive_daily_cap(RECORDED_DAILY_READS)
    live_look = EffectPosterior(mean=0.696, variance=0.0021, n=119)   # live 2026-10-21
    path, why = decide(ref, live_look, 679.3, floor)
    return [
        ("floor_unchanged_0.02", floor == 0.02),
        ("cap_default_is_derived", derived == pull.DERIVED_DAILY_CAP == settings_cap == env_cap == 8),
        ("route_horizon_12000_queue_object", (ref.horizon_sec, ref.object_source) == (12000.0, "queue")),
        ("painting_never_held_back", act_episodes.holdback_rule("dispatch")[0] is False),
        ("reading_pull_held_back", act_episodes.holdback_rule("queue")[0] is True),
        ("live_look_admitted_on_proven_path", (path, why) == ("proven_effect", "admitted")),
        ("live_look_below_info_floor",
         decide(ref, live_look, 679.3, floor, n_control=0)[0] != "information"),
    ]


def simulate(ticks, *, post_at, cost_at, ref, floor, arousal_at, cap) -> dict:
    """One reading-pull candidate per tick, in time order, through the worker's rules."""
    out = {"decisions": {}, "refused_before_allocator": {}, "allocator_refused": {}, "check_rpcs": 0,
           "doer_refusals": {}, "control": 0, "treated_started": 0, "cap_hits": 0, "by_day": {}}
    busy_until = hold_until = None
    started_by_day: dict[str, int] = {}
    for t, dispatch_id in ticks:
        if hold_until is not None and t < hold_until:
            out["refused_before_allocator"]["doer_refused_hold"] = out["refused_before_allocator"].get(
                "doer_refused_hold", 0) + 1
            continue
        if busy_until is not None and t < busy_until:
            out["refused_before_allocator"]["act_in_flight"] = out["refused_before_allocator"].get("act_in_flight", 0) + 1
            continue
        path, why = decide(ref, post_at(t), cost_at(t), floor)
        if path == "none":
            out["allocator_refused"][why] = out["allocator_refused"].get(why, 0) + 1
            continue
        out["decisions"][path] = out["decisions"].get(path, 0) + 1
        # Hub check (gates in order). Kill switch on, Wallet A window 0-0 (live), refund backoff not
        # replayable (Redis only) -- treated as open, disclosed.
        day = t.date().isoformat()
        out["check_rpcs"] += 1
        verdict = pull.pull_block_reason(pull.PullGateInputs(
            enabled=True, wallet_reason=None, wallet_retry_sec=None, pulled_today=started_by_day.get(day, 0),
            daily_cap=cap, arousal_level=arousal_at(t), thermal=None, now=t))
        if not verdict.open:
            out["doer_refusals"][verdict.refusal] = out["doer_refusals"].get(verdict.refusal, 0) + 1
            out["cap_hits"] += verdict.refusal == "daily_cap"
            hold_until = t + timedelta(seconds=float(verdict.retry_after_sec or 0.0))
            continue
        arm = "control" if path == "proven_effect" and stable_draw(dispatch_id) < FRACTION else "treated"
        if arm == "control":
            out["control"] += 1
        else:
            out["treated_started"] += 1
            started_by_day[day] = started_by_day.get(day, 0) + 1
        out["by_day"].setdefault(day, {"treated_started": 0, "control": 0})[
            "control" if arm == "control" else "treated_started"] += 1
        busy_until = t + timedelta(seconds=float(ref.horizon_sec))
    return out


def _arousal_now() -> dict | None:
    url = os.environ.get("ORION_BUS_URL")
    if not url:
        return None
    try:
        import redis

        raw = redis.Redis.from_url(url, socket_timeout=3).get("orion:regulation:latest")
        return (json.loads(raw) or {}).get("arousal") if raw else None
    except Exception:  # noqa: BLE001
        return None


def live(hours: float) -> dict:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    dsn = os.environ.get("ORION_PG_DSN")
    if not dsn:
        raise SystemExit("ORION_PG_DSN required for --live")
    floor = settings_floor()
    ref = _read_ref()
    report: dict = {"hours": hours, "floor_nats_per_sec": floor}
    with psycopg2.connect(dsn, options="-c default_transaction_read_only=on -c statement_timeout=120000") as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT now() AS now")
        now = cur.fetchone()["now"]
        since = now - timedelta(hours=hours)

        # 1. derivations
        cur.execute(pull.DAILY_READS_STARTED_SQL.replace("$1", "%s"), (30,))
        series = [int(r["n"]) for r in cur.fetchall()]
        cur.execute("""
            WITH d AS (SELECT seed_id, min(created_at) sub FROM reading_durable_turn WHERE stage = 1 GROUP BY 1)
            SELECT count(*) AS n,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM a.closed_at - d.sub)) AS p50,
                   percentile_cont(0.9) WITHIN GROUP (ORDER BY extract(epoch FROM a.closed_at - d.sub)) AS p90
              FROM grammar_acts a JOIN d ON d.seed_id = substr(a.test_before, length('read_seed:') + 1)
             WHERE a.actor = 'hub.world_pulse_read' AND a.outcome_value = 1 AND a.closed_at IS NOT NULL""")
        h = cur.fetchone()
        report["derivations"] = {
            "daily_reads_30d": series, "derived_daily_cap": pull.derive_daily_cap(series),
            "shipped_daily_cap": pull.DERIVED_DAILY_CAP,
            "horizon_basis": {"n": h["n"], "p50_sec": round(float(h["p50"] or 0)), "p90_sec": round(float(h["p90"] or 0))},
            "shipped_horizon_sec": ref.horizon_sec}

        # look posterior and cost, time-correct
        cur.execute("""SELECT outcome_value, closed_at, extract(epoch FROM closed_at - opened_at) AS work
                         FROM grammar_acts WHERE actor = %s AND verb = %s AND signal_id = %s
                          AND closed_at IS NOT NULL AND outcome_value IS NOT NULL ORDER BY closed_at""",
                    (ref.actor, ref.verb, ref.signal_id))
        closes = [(_aware(r["closed_at"]), float(r["outcome_value"]), float(r["work"] or 0)) for r in cur.fetchall()]

        def post_at(t):
            return rebuild_posterior([v for at, v, _ in closes if at <= t])

        def cost_at(t):
            w = [s for at, _, s in closes if t - COST_LOOKBACK < at <= t and s > 0]
            return median(w) if w else None

        # 2. tick population
        cur.execute("""
            SELECT DISTINCT f.generated_at, f.frame_id
              FROM substrate_execution_dispatch_frames f,
                   jsonb_array_elements((f.dispatch_frame_json->'blocked_candidates') || (f.dispatch_frame_json->'dispatched_candidates')) c
             WHERE f.created_at > %s AND c->>'cortex_verb' = %s
               AND coalesce(jsonb_typeof(c->'visual_baseline'), 'null') = 'null'
             ORDER BY f.generated_at""", (since, RENDER_VERB))
        ticks = [(_aware(r["generated_at"]), f"dispatch:proposal:read_world_pulse:{r['frame_id']}") for r in cur.fetchall()]
        cur.execute("""SELECT reason, count(*) AS n FROM endogenous_outreach_decisions
                        WHERE decided_at > %s GROUP BY 1 ORDER BY 2 DESC""", (since,))
        report["arousal_cross_check_outreach_decisions"] = {r["reason"]: r["n"] for r in cur.fetchall()}

    arousal = _arousal_now()
    strained_since = None
    if arousal and arousal.get("arousal_level") == "strained" and arousal.get("since"):
        strained_since = datetime.fromisoformat(str(arousal["since"]).replace("Z", "+00:00"))
    report["arousal_now"] = {k: arousal.get(k) for k in ("arousal_level", "since", "reasons", "cabinet_temp_c")} if arousal else None

    def lived(t):
        # The current level holds since `since`; before it the level is not recorded: unknown.
        return "strained" if strained_since is not None and t >= strained_since else None

    report["ticks"] = len(ticks)
    report["as_lived"] = simulate(ticks, post_at=post_at, cost_at=cost_at, ref=ref, floor=floor,
                                  arousal_at=lived, cap=pull.DERIVED_DAILY_CAP)
    report["counterfactual_arousal_calm"] = simulate(ticks, post_at=post_at, cost_at=cost_at, ref=ref, floor=floor,
                                                     arousal_at=lambda t: None, cap=pull.DERIVED_DAILY_CAP)
    report["painting_holdback"] = {"rule": act_episodes.holdback_rule("dispatch"),
                                   "control_draws_possible": 0}
    report["unreplayable"] = ["thermal (no cabinet temperature history in Postgres)",
                              "Wallet A refund backoff (Redis only)",
                              "seed availability per tick (queue held 25+ pending seeds throughout; assumed non-empty)"]
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--hours", type=float, default=48.0)
    args = ap.parse_args()
    checks = fixture_checks()
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    if args.live:
        print(json.dumps(live(args.hours), indent=2, default=str))
    passed = sum(ok for _, ok in checks)
    print(f"{passed}/{len(checks)} PASS")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
