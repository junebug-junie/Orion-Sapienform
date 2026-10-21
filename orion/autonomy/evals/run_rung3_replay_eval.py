#!/usr/bin/env python3
"""Thinking rung 3 replay eval: bind the winner to action, and compete across doers.

Design: docs/superpowers/specs/2026-10-14-orion-thinking-architecture-design.md, rung 3.
Proof asked for: "allocator admissions from real candidates with an unchanged floor, plus treated
episodes". This eval replays real rows through the SAME functions the runtime now uses, and reports
what they decide -- including when the honest answer is "nothing is admitted".

Fixture mode (CI, default): deterministic checks on the arithmetic and the rules --
  * the floor read from the dispatch runtime's settings is still 0.02 nats/s;
  * a reconstructed act posterior equals the act learner's own sequential update;
  * a measured act at its measured cost is refused at the floor, a never-graded one is admitted
    once, an act with no measured cost is refused, and the largest cost each live act could carry
    and still clear the floor is reported (nats / floor).

Live mode (``--live``, needs ORION_PG_DSN; read-only transaction, statement timeout). For the last
``--hours`` (48):
  1. binding: broadcast ticks where an eligible world-first winner was dropped to "none" (now
     floored to defer) and ticks attended by a world: ref (now bindable);
  2. act posteriors rebuilt from grammar_acts closes, checked against the stored posterior table;
  3. every render_scene candidate the dispatch runtime saw, re-scored as the painting act at the
     unchanged floor with the time-correct posterior and 30-day measured cost: old verdict vs new;
  4. the cross-doer table: every rung-2 act cell, its nats, its measured cost, nats/s, verdict at the
     floor, and the cost ceiling under which it would clear;
  5. holdback: acting ticks the per-tick draw may now take (field-signal candidates sent), and the
     world-episode arms (control / treated) recorded in the window;
  6. resolver: chat-reply acts that open under a bindable world:chat winner, i.e. the actor='orion'
     acted verdicts the writer will now write.

    python orion/autonomy/evals/run_rung3_replay_eval.py
    ORION_PG_DSN=... python orion/autonomy/evals/run_rung3_replay_eval.py --live --hours 48
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orion.autonomy.act_candidates import ActRef, act_allocator_candidate  # noqa: E402
from orion.autonomy.allocator import allocate, expected_information_gain_nats  # noqa: E402
from orion.autonomy.prediction import (  # noqa: E402
    ACT_BINARY_OBSERVATION_VARIANCE,
    EffectPosterior,
    cold_act_posterior,
    score_act_outcome,
)
from orion.autonomy.self_shed import bind_workspace_winner  # noqa: E402

SETTINGS = ROOT / "services/orion-execution-dispatch-runtime/app/settings.py"
RENDER_VERB = "skills.imagination.render_scene.v1"
PAINT = ActRef(verb="make", actor="thought.painting", signal_id="artifact_produced")
ROUTE_COST_KEY = ("express", "host:circe_gpu")
ACT_COST_LOOKBACK = timedelta(days=30)
CHAT_REFS = ("world:chat", "node:substrate.chat")


def settings_floor() -> float:
    """The floor as shipped (code default of ORION_DISPATCH_MIN_NATS_PER_SEC), read, not assumed."""
    text = SETTINGS.read_text()
    m = re.search(r"orion_dispatch_min_nats_per_sec: float = Field\(\s*([0-9.]+)", text)
    if not m:
        raise SystemExit("could not read ORION_DISPATCH_MIN_NATS_PER_SEC default from settings.py")
    return float(m.group(1))


def rebuild_posterior(values: list[float]) -> EffectPosterior:
    post = cold_act_posterior()
    for v in values:
        post, _, _ = score_act_outcome(post, v)
    return post


def verdict(ref: ActRef, post: EffectPosterior | None, cost: float | None, floor: float) -> str:
    posteriors = {} if post is None or post.n == 0 else {ref.cell: post}
    cand = act_allocator_candidate(dispatch_id="x", dispatch_kind="k", target_id="t", ref=ref,
                                   posteriors=posteriors, cost_sec=cost)
    out = allocate([cand], allowance_sec=1e9, min_nats_per_sec=floor)
    return "admitted" if out.admitted else out.refused[0][1]


def cost_ceiling(post: EffectPosterior, floor: float) -> float:
    return expected_information_gain_nats(post.variance, ACT_BINARY_OBSERVATION_VARIANCE) / floor


def fixture_checks() -> list[tuple[str, bool]]:
    floor = settings_floor()
    checks = [("floor_unchanged_0.02", floor == 0.02)]
    seq = [0.0] * 28 + [1.0]
    post = rebuild_posterior(seq)
    manual = cold_act_posterior()
    for v in seq:
        manual, _, _ = score_act_outcome(manual, v)
    checks.append(("rebuild_matches_act_learner", (post.mean, post.variance, post.n) == (manual.mean, manual.variance, manual.n)))
    live_paint = EffectPosterior(mean=0.017, variance=0.0083, n=29)  # live 2026-10-21
    checks.append(("measured_paint_refused_at_floor", verdict(PAINT, live_paint, 57.1, floor) == "below_information_floor"))
    checks.append(("cold_act_admitted_once", verdict(PAINT, None, 57.1, floor) == "admitted"))
    checks.append(("no_cost_refused", verdict(PAINT, live_paint, None, floor) == "no_cost_estimate"))
    # One observation in, a binary act clears 0.02 nats/s only if it costs under ~10 s.
    one = rebuild_posterior([1.0])
    checks.append(("one_obs_ceiling_about_10s", 9.0 < cost_ceiling(one, floor) < 11.0))
    return checks


# --- live --------------------------------------------------------------------------------------

def _aware(ts):
    return ts if ts is None or ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def live(hours: float) -> dict:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    dsn = os.environ.get("ORION_PG_DSN") or os.environ.get("POSTGRES_URI")
    if not dsn:
        raise SystemExit("ORION_PG_DSN required for --live")
    floor = settings_floor()
    report: dict = {"hours": hours, "floor_nats_per_sec": floor}
    with psycopg2.connect(dsn, options="-c default_transaction_read_only=on -c statement_timeout=120000") as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT now() AS now")
        now = cur.fetchone()["now"]
        since = now - timedelta(hours=hours)

        # 1. binding
        cur.execute(
            """
            SELECT count(*) AS ticks,
                   count(*) FILTER (WHERE projection_json->>'selected_action_type' = 'none') AS none_ticks,
                   count(*) FILTER (WHERE projection_json->>'selected_action_type' = 'none'
                       AND projection_json#>>'{frame,debug,world_first,winner}' IS NOT NULL
                       AND jsonb_array_length(coalesce(projection_json#>'{frame,candidate_actions}','[]')) > 0)
                       AS eligible_winner_dropped,
                   count(*) FILTER (WHERE projection_json->'attended_node_ids'->>0 LIKE 'world:%%') AS world_ref_ticks
              FROM substrate_attention_broadcast_log WHERE generated_at > %s
            """, (since,))
        report["binding"] = dict(cur.fetchone())
        cur.execute(
            """
            SELECT count(*) AS frames,
                   count(*) FILTER (WHERE EXISTS (SELECT 1 FROM jsonb_array_elements_text(proposal_frame_json->'warnings') w
                                                  WHERE w LIKE 'winner_unbindable:no_selected_action%%')) AS no_selected_action
              FROM substrate_proposal_frames WHERE created_at > %s
            """, (since,))
        report["binding"]["proposal_frames"] = dict(cur.fetchone())

        # 2. act posteriors rebuilt from closes vs the stored table
        cur.execute(
            """
            SELECT actor, verb, signal_id, outcome_value, closed_at,
                   extract(epoch FROM closed_at - opened_at) AS work_sec, outcome_source
              FROM grammar_acts WHERE closed_at IS NOT NULL AND outcome_value IS NOT NULL
             ORDER BY closed_at
            """)
        closes = cur.fetchall()
        cur.execute(
            "SELECT dispatch_kind, target_id, signal_id, posterior_mean, posterior_variance, posterior_n "
            "FROM substrate_action_effect_posterior WHERE dispatch_kind LIKE 'act:%%' AND baseline_bin = 0")
        stored = {(r["dispatch_kind"], r["target_id"], r["signal_id"]): r for r in cur.fetchall()}
        cells: dict[tuple[str, str, str], list] = {}
        for r in closes:
            cells.setdefault((f"act:{r['verb']}", r["actor"], r["signal_id"]), []).append(r)
        rebuilt = {}
        for key, rows in cells.items():
            post = rebuild_posterior([float(r["outcome_value"]) for r in rows])
            s = stored.get(key)
            rebuilt[key] = post
            rebuilt_row = {"rebuilt_n": post.n, "rebuilt_mean": round(post.mean, 4),
                           "stored_n": s and s["posterior_n"], "stored_mean": s and round(float(s["posterior_mean"]), 4)}
            rebuilt_row["match"] = bool(s) and s["posterior_n"] == post.n and abs(float(s["posterior_mean"]) - post.mean) < 1e-6
            report.setdefault("posterior_rebuild", {})["|".join(key)] = rebuilt_row

        def post_at(key, t):
            return rebuild_posterior([float(r["outcome_value"]) for r in cells.get(key, []) if _aware(r["closed_at"]) <= t])

        # 3. every render_scene candidate the runtime saw, re-scored as the painting act
        cur.execute(
            "SELECT created_at, latency_ms FROM substrate_dispatch_results WHERE dispatch_kind = %s AND target_id = %s "
            "AND latency_ms IS NOT NULL AND latency_ms > 0 AND created_at > %s ORDER BY created_at",
            (*ROUTE_COST_KEY, since - ACT_COST_LOOKBACK))
        costs = [(r["created_at"], float(r["latency_ms"]) / 1000.0) for r in cur.fetchall()]
        cur.execute(
            """
            SELECT f.generated_at, c->>'dispatch_id' AS dispatch_id, (c->'visual_baseline') IS NOT NULL
                   AND c->>'visual_baseline' <> 'null' AS baseline,
                   (SELECT string_agg(r, ',') FROM jsonb_array_elements_text(c->'reasons') r WHERE r LIKE 'allocator:%%') AS old
              FROM substrate_execution_dispatch_frames f,
                   jsonb_array_elements((f.dispatch_frame_json->'blocked_candidates') || (f.dispatch_frame_json->'dispatched_candidates')) c
             WHERE f.created_at > %s AND c->>'cortex_verb' = %s
            """, (since, RENDER_VERB))
        render = cur.fetchall()
        key = (f"act:{PAINT.verb}", PAINT.actor, PAINT.signal_id)
        tally: dict[str, int] = {}
        old_tally: dict[str, int] = {}
        for r in render:
            t = _aware(r["generated_at"])
            window = [c for at, c in costs if t - ACT_COST_LOOKBACK < _aware(at) <= t]
            cost = median(window) if window else None
            new = "baseline_lane_unchanged" if r["baseline"] else verdict(PAINT, post_at(key, t), cost, floor)
            tally[new] = tally.get(new, 0) + 1
            old_tally[r["old"] or "not_refused_by_allocator"] = old_tally.get(r["old"] or "not_refused_by_allocator", 0) + 1
        report["render_scene_candidates"] = {"n": len(render), "old_verdicts": old_tally, "new_verdicts": tally}

        # 4. cross-doer table at window end (one currency; informational for doers with no dispatch seam)
        dispatchable = {PAINT.actor: "render_scene dispatch (all doer gates kept)"}
        table = []
        for (kind, actor, signal), rows in sorted(cells.items()):
            post = rebuilt[(kind, actor, signal)]
            if actor == PAINT.actor:
                window = [c for at, c in costs if now - ACT_COST_LOOKBACK < _aware(at) <= now]
                cost, cost_src = (median(window) if window else None), "route p50 latency, 30 d"
            else:
                work = [float(r["work_sec"]) for r in rows if r["outcome_source"] == "producer" and r["work_sec"]]
                cost, cost_src = (median(work) if work else None), ("producer open->close p50" if work else "no measured work time")
            nats = expected_information_gain_nats(post.variance, ACT_BINARY_OBSERVATION_VARIANCE)
            ref = ActRef(verb=kind.split(":", 1)[1], actor=actor, signal_id=signal)
            table.append({
                "act": f"{kind}|{actor}|{signal}", "n": post.n, "mean": round(post.mean, 3),
                "nats": round(nats, 4), "cost_sec": None if cost is None else round(cost, 1), "cost_source": cost_src,
                "nats_per_sec": None if not cost else round(nats / cost, 6),
                "verdict_at_floor": verdict(ref, post, cost, floor),
                "max_cost_to_clear_floor_sec": round(cost_ceiling(post, floor), 2),
                "dispatch_seam": dispatchable.get(actor, "none (doer self-schedules)"),
            })
        report["cross_doer"] = table

        # 5. holdback
        cur.execute(
            """
            SELECT count(*) FILTER (WHERE EXISTS (
                       SELECT 1 FROM jsonb_array_elements(f.dispatch_frame_json->'dispatched_candidates') c
                        WHERE c->'expected_effect' IS NOT NULL AND c->>'expected_effect' <> 'null'
                          AND (c->'world_action' IS NULL OR c->>'world_action' = 'null')
                          AND (c->'visual_baseline' IS NULL OR c->>'visual_baseline' = 'null'))) AS field_signal_acting_ticks,
                   count(*) FILTER (WHERE (f.dispatch_frame_json->>'dispatch_count')::int > 0) AS acting_ticks
              FROM substrate_execution_dispatch_frames f WHERE f.created_at > %s
            """, (since,))
        hb = dict(cur.fetchone())
        cur.execute("SELECT arm, count(*) AS n, max(decided_at) AS last FROM substrate_world_action_episodes GROUP BY arm")
        hb["world_episode_arms_all_time"] = {r["arm"]: {"n": r["n"], "last": r["last"].isoformat()} for r in cur.fetchall()}
        report["holdback"] = hb

        # 6. resolver: chat replies that would be bound to a world:chat winner
        cur.execute("SELECT act_id, opened_at FROM grammar_acts WHERE actor = 'hub.chat_reply' AND opened_at IS NOT NULL")
        replies = cur.fetchall()
        bound = 0
        for r in replies:
            at = _aware(r["opened_at"])
            cur.execute(
                "SELECT log_id, generated_at, projection_json->>'selected_action_type' AS a, "
                "projection_json->>'selected_open_loop_id' AS l, projection_json->'attended_node_ids' AS n, "
                "projection_json->>'dwell_ticks' AS d FROM substrate_attention_broadcast_log "
                "WHERE generated_at <= %s AND generated_at > %s ORDER BY generated_at DESC LIMIT 1",
                (at, at - timedelta(seconds=90)))
            b = cur.fetchone()
            if b is None:
                continue
            proj = {"selected_action_type": b["a"], "selected_open_loop_id": b["l"], "generated_at": _aware(b["generated_at"]),
                    "dwell_ticks": int(b["d"] or 0), "attended_node_ids": list(b["n"] or [])}
            winner, _ = bind_workspace_winner(proj, broadcast_log_id=b["log_id"], binds_to_nodes=CHAT_REFS, now=at)
            bound += winner is not None
        cur.execute("SELECT count(*) AS n FROM attention_loop_outcome WHERE actor = 'orion' AND created_at > %s", (since,))
        report["resolver"] = {"chat_reply_acts_all_time": len(replies), "would_bind": bound,
                              "orion_verdicts_in_window_today": cur.fetchone()["n"]}
    report["admissions_at_unchanged_floor"] = report["render_scene_candidates"]["new_verdicts"].get("admitted", 0)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--hours", type=float, default=48.0)
    args = ap.parse_args()
    if args.live:
        print(json.dumps(live(args.hours), indent=2, default=str))
        return 0
    checks = fixture_checks()
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
