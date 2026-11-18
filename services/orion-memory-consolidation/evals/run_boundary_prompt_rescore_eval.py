#!/usr/bin/env python3
"""Re-score 30 days of real turn pairs with the classify prompt before and after BOUNDARY got a definition.

    python services/orion-memory-consolidation/evals/run_boundary_prompt_rescore_eval.py \
        --summary-json services/orion-memory-consolidation/evals/results/2026-10-06-boundary-prompt-rescore-summary.json

What it does (read-only on Postgres; the model is called through the live LLM gateway over the bus):

* Loads Juniper's direct chat turns of the last ``--days`` days (full text, held in memory only).
* For each turn after the first, builds the exact first-pass classify prompt the live service builds
  (baseline = the previous turn, ``phase`` = what the live prompt sees, i.e. ``unknown`` while
  MEMORY_LEGACY_BOUNDARY_USE_PHASE is off) twice: ``before`` = the prompt without
  ``BOUNDARY_DEFINITION``, ``after`` = the current prompt. Same route and options as
  ``app/classify.py::_llm_classify``.
* Parses the BOUNDARY score with the live parser (``app.boundary.scores_from_llm_result``).
* Replays Rule 3 (wall-clock phase from the gap to the previous turn + the score) with each set of
  scores and reports: score distribution, the resumed_thread >= 0.92 rate, episodes per day, and the
  episodes covering the 2026-10-05 02:55-04:20 UTC hub_orion session (the Chicago session) and the
  2026-09-28 Austin morning.

PRIVACY: the summary holds only scores, counts, phases and times. No prompt or response text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parent
REPO_ROOT = HERE.parents[2]
for p in (REPO_ROOT, SERVICE, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from orion.memory.turn_change_classify import BOUNDARY_DEFINITION, build_turn_change_prompt  # noqa: E402
from orion.situational.context import classify_conversation_phase  # noqa: E402

PG_CONTAINER = "orion-athena-sql-db"
TZ = "America/Denver"
LOCAL = ZoneInfo(TZ)
DEFAULT_BUS = os.environ.get("ORION_BUS_URL", "redis://100.92.216.81:6379/0")
OVERRIDE_THRESHOLD = 0.92
FALLBACK_GAP_SEC = 5400
REORIENT = {"long_gap", "next_day", "stale_thread"}
ACTIVE = {"same_breath", "short_pause"}
CLIP = 300  # app/classify.py::_MAX_TURN_FIELD_CHARS

SESSIONS = {
    "chicago_2026_10_05": ("2026-10-05T02:50:00+00:00", "2026-10-05T04:25:00+00:00"),
    "austin_2026_09_28": ("2026-09-28T06:00:00+00:00", "2026-09-28T10:30:00+00:00"),
}


def _psql_json(sql: str) -> Any:
    out = subprocess.run(
        ["docker", "exec", PG_CONTAINER, "psql", "-U", "postgres", "-d", "conjourney", "-v", "ON_ERROR_STOP=1",
         "-Atc", f"BEGIN READ ONLY; {sql}; ROLLBACK;"],
        capture_output=True, text=True, timeout=120, check=True,
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.strip() and ln not in {"BEGIN", "ROLLBACK"}]
    return json.loads(lines[-1])


def load_turns(days: int) -> list[dict[str, Any]]:
    sql = f"""
    SELECT coalesce(json_agg(json_build_object(
        'correlation_id', correlation_id, 'prompt', prompt, 'response', response,
        'at', to_char(created_at, 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
        'temporal_phase', spark_meta->>'temporal_phase',
        'live_score', (spark_meta->>'conversation_boundary_score')::float) ORDER BY created_at), '[]'::json)
    FROM chat_history_log
    WHERE created_at > now() - interval '{int(days)} days'
      AND coalesce(trim(prompt), '') <> '' AND coalesce(trim(response), '') <> ''
      AND client_meta->'external_room' IS NULL
    """
    return _psql_json(sql)


def _ts(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _clip(s: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= CLIP else s[: CLIP - 3] + "..."


def prompts_for(turn: dict[str, Any], prev: dict[str, Any]) -> tuple[str, str]:
    """(before, after): the live first-pass prompt without and with the BOUNDARY definition."""
    baseline = f"User: {_clip(prev['prompt'])}\nOrion: {_clip(prev['response'])}\n"
    after = build_turn_change_prompt(
        prompt=turn["prompt"], response=turn["response"], baseline_mode="prior_turn",
        baseline_text=baseline, phase=str(turn.get("temporal_phase") or "unknown"),
    )
    assert BOUNDARY_DEFINITION in after
    return after.replace(BOUNDARY_DEFINITION, ""), after


async def classify(bus, prompt: str, *, route: str, timeout: float) -> dict[str, Any]:
    from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
    from app.boundary import scores_from_llm_result

    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)], route=route,
        options={"return_logprobs": True, "logprobs_top_k": 8, "logprob_summary_only": False, "max_tokens": 24,
                 "llm_route": route, "purpose": "classify", "skip_spark_candidate_publish": True,
                 "chat_template_kwargs": {"enable_thinking": False}, "gateway_read_timeout_sec": timeout})
    corr = str(uuid4())
    reply = f"orion:exec:result:LLMGatewayService:{corr}"
    env = BaseEnvelope(kind="llm.chat.request", correlation_id=corr, reply_to=reply,
                       source=ServiceRef(name="boundary-prompt-eval", version="0", node="athena"),
                       payload=payload.model_dump(mode="json"))
    msg = await bus.rpc_request("orion:exec:request:LLMGatewayService", env, reply_channel=reply,
                                timeout_sec=timeout)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error)
    result = decoded.envelope.payload or {}
    content = str(result.get("content") or result.get("text") or "")
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    return scores_from_llm_result(content, raw)


def annotate(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prev_at = None
    out = []
    for t in turns:
        at = _ts(t["at"])
        clock = classify_conversation_phase(prev_at, at, TZ)
        out.append({**t, "at_dt": at, "phase": clock["phase_change"],
                    "gap_sec": (at - prev_at).total_seconds() if prev_at else None})
        prev_at = at
    return out


def rule3(phase: str, score: float | None, gap: float | None) -> bool:
    if phase in REORIENT:
        return True
    if phase == "resumed_thread":
        return score is not None and score >= OVERRIDE_THRESHOLD
    if phase in ACTIVE:
        return False
    return gap is not None and gap >= FALLBACK_GAP_SEC


def split(turns: list[dict[str, Any]], key: str) -> list[list[dict[str, Any]]]:
    episodes: list[list[dict[str, Any]]] = []
    for t in turns:
        if episodes and rule3(t["phase"], t.get(key), t["gap_sec"]):
            episodes.append([t])
        elif episodes:
            episodes[-1].append(t)
        else:
            episodes.append([t])
    return episodes


LEGACY_LLM_ONLY_THRESHOLD = 0.85  # settings.MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD


def legacy_live_windows(turns: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """The LIVE window rule as it runs today (worker.py -> window_fetch.legacy_close_decision).

    legacy_view() hides the phase stamp, so every turn is phase "unknown": the window closes when
    the boundary score is >= 0.85 (``legacy:unknown_phase+llm``), else when the gap between the
    window's last two turns is >= 5400 s (``legacy:time_gap``). The closing turn ends this window and
    seeds the next (WindowStore.close_current_window). Each close runs the live crystallization intake.
    """
    windows: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for t in turns:
        cur.append(t)
        score = t.get(key)
        reason = None
        if isinstance(score, (int, float)) and score >= LEGACY_LLM_ONLY_THRESHOLD:
            reason = "llm"
        elif len(cur) >= 2 and (cur[-1]["at_dt"] - cur[-2]["at_dt"]).total_seconds() >= FALLBACK_GAP_SEC:
            reason = "time_gap"
        if reason and len(cur) > 1:
            windows.append({"turns": cur, "reason": reason})
            cur = [t]
    if cur:
        windows.append({"turns": cur, "reason": None})
    return windows


def legacy_summary(turns: list[dict[str, Any]], key: str) -> dict[str, Any]:
    ws = legacy_live_windows(turns, key)
    closed = [w for w in ws if w["reason"]]
    sizes = [len(w["turns"]) for w in ws]
    return {
        "windows": len(ws),
        "closed": len(closed),
        "closed_by": dict(Counter(w["reason"] for w in closed)),
        "turns_per_window_mean": round(statistics.mean(sizes), 2) if sizes else None,
        "turns_per_window_median": statistics.median(sizes) if sizes else None,
    }


_SIGNIFICANT = {"TOPIC", "STANCE", "REPAIR"}


def consumer_flags(sc: dict[str, Any]) -> dict[str, bool]:
    """Per-turn decisions the live consumers of the SAME classify call make from the first pass.

    Thresholds are the live defaults: turn_change signal (worker._maybe_publish_turn_change_signal:
    novelty >= 0.65 and confidence >= 0.15), consolidation_gate substantive_shift (shift in
    TOPIC/STANCE/REPAIR and novelty >= 0.35), recall_skip_gate (novelty < 0.25 floor; a significant
    shift with novelty >= 0.35 blocks the skip), retrieval_intent (REPAIR -> open_loop, STANCE ->
    relational, both at novelty >= 0.35), should_session_reappraise (margin 0.15), the
    consolidation gate's significance floor (MEMORY >= 0.40), and the live window close (>= 0.85).
    """
    from orion.memory.turn_change_classify import should_session_reappraise

    nov = sc.get("novelty_score")
    n = float(nov) if isinstance(nov, (int, float)) else None
    shift = str(sc.get("shift_kind") or "NONE").upper()
    conf = sc.get("confidence")
    mem = sc.get("memory_significance_score")
    bnd = sc.get("conversation_boundary_score")
    return {
        "turn_change_signal_emitted": n is not None and n >= 0.65 and isinstance(conf, (int, float)) and conf >= 0.15,
        "substantive_shift": shift in _SIGNIFICANT and n is not None and n >= 0.35,
        "recall_novelty_below_floor": n is None or n < 0.25,
        "retrieval_intent_open_loop": shift == "REPAIR" and n is not None and n >= 0.35,
        "retrieval_intent_relational": shift == "STANCE" and n is not None and n >= 0.35,
        "session_reappraise": should_session_reappraise(sc, margin=0.15, prior_turn_count=1),
        "memory_significance_ge_0_40": isinstance(mem, (int, float)) and mem >= 0.40,
        "live_window_close_ge_0_85": isinstance(bnd, (int, float)) and bnd >= LEGACY_LLM_ONLY_THRESHOLD,
    }


def other_lines_report(turns: list[dict[str, Any]], runs: int) -> dict[str, Any]:
    """NOVEL / SHIFT / MEMORY before vs after on the same turns, with run-to-run noise as the yardstick."""
    keys = [f"{p}_r{i}" for p in ("before", "after") for i in range(runs)]
    scored = [t for t in turns if all(k in (t.get("scores") or {}) for k in keys)]
    per_run = {}
    for k in keys:
        sc = [t["scores"][k] for t in scored]
        per_run[k] = {
            "novelty": distribution([x["novelty_score"] for x in sc if isinstance(x.get("novelty_score"), (int, float))]),
            "memory": distribution([x["memory_significance_score"] for x in sc
                                    if isinstance(x.get("memory_significance_score"), (int, float))]),
            "shift_counts": dict(Counter(str(x.get("shift_kind") or "missing") for x in sc)),
            "consumer_true_counts": dict(Counter(f for x in sc for f, v in consumer_flags(x).items() if v)),
        }

    def flips(a: str, b: str) -> dict[str, int]:
        out = Counter()
        for t in scored:
            fa, fb = consumer_flags(t["scores"][a]), consumer_flags(t["scores"][b])
            for f in fa:
                out[f] += int(fa[f] != fb[f])
            out["shift_kind"] += int(str(t["scores"][a].get("shift_kind")) != str(t["scores"][b].get("shift_kind")))
        return dict(out)

    comparisons = {"before_vs_after": flips("before_r0", "after_r0")}
    if runs >= 2:
        comparisons["noise_before_r0_vs_r1"] = flips("before_r0", "before_r1")
        comparisons["noise_after_r0_vs_r1"] = flips("after_r0", "after_r1")
        comparisons["before_vs_after_r1"] = flips("before_r1", "after_r1")
    return {"turns_scored_every_run": len(scored), "per_run": per_run, "per_turn_flips": comparisons}


def distribution(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"n": 0}
    bins = Counter(min(int(s * 10), 9) for s in scores)
    return {
        "n": len(scores),
        "mean": round(statistics.mean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "share_ge_0_92": round(sum(s >= OVERRIDE_THRESHOLD for s in scores) / len(scores), 3),
        "share_ge_0_5": round(sum(s >= 0.5 for s in scores) / len(scores), 3),
        "deciles": {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": bins.get(b, 0) for b in range(10)},
    }


def session_view(episodes: list[list[dict[str, Any]]], window: tuple[str, str], key: str) -> dict[str, Any]:
    a, b = (_ts(x) for x in window)
    hit = [ep for ep in episodes if any(a <= t["at_dt"] <= b for t in ep)]
    turns = [t for ep in hit for t in ep if a <= t["at_dt"] <= b]
    return {
        "episodes": len(hit),
        "turns_in_window": len(turns),
        "episode_sizes_in_window": [sum(1 for t in ep if a <= t["at_dt"] <= b) for ep in hit],
        "turn_scores": [
            {"utc": t["at_dt"].strftime("%m-%d %H:%M"), "phase": t["phase"],
             "score": None if t.get(key) is None else round(t[key], 3)}
            for t in turns
        ],
    }


def report(turns: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"turns": len(turns), "phase_histogram": dict(Counter(t["phase"] for t in turns))}
    resumed = [t for t in turns if t["phase"] == "resumed_thread"]
    for key in ("before", "after"):
        scored = [t[key] for t in turns if t.get(key) is not None]
        rs = [t[key] for t in resumed if t.get(key) is not None]
        eps = split(turns, key)
        per_day = Counter(ep[0]["at_dt"].astimezone(LOCAL).date().isoformat() for ep in eps)
        # Label-free coherence: does the judge's BOUNDARY agree with its own SHIFT line and with the
        # wall clock? A definition that only pushed every score to 0 would flatten these splits too.
        by_shift = {k: distribution([t[key] for t in turns if t.get(f"{key}_shift") == k and t.get(key) is not None])
                    for k in sorted({str(t.get(f"{key}_shift")) for t in turns if t.get(f"{key}_shift")})}
        by_phase = {ph: distribution([t[key] for t in turns if t["phase"] == ph and t.get(key) is not None])
                    for ph in sorted({t["phase"] for t in turns})}
        out[key] = {
            "score_distribution": distribution(scored),
            "score_by_shift": {k: {f: v.get(f) for f in ("n", "mean", "share_ge_0_5")} for k, v in by_shift.items()},
            "score_by_phase": {k: {f: v.get(f) for f in ("n", "mean", "share_ge_0_5")} for k, v in by_phase.items()},
            "unscored_turns": sum(1 for t in turns[1:] if t.get(key) is None),
            "resumed_thread_scored": len(rs),
            "resumed_thread_ge_0_92": sum(s >= OVERRIDE_THRESHOLD for s in rs),
            "episodes": len(eps),
            "active_days": len(per_day),
            "episodes_per_active_day_mean": round(len(eps) / len(per_day), 2) if per_day else None,
            "episodes_per_day": dict(sorted(per_day.items())),
            "sessions": {name: session_view(eps, win, key) for name, win in SESSIONS.items()},
            "live_legacy_windows": legacy_summary(turns, key),
        }
    runs = max((int(k.rsplit("_r", 1)[1]) + 1 for t in turns for k in (t.get("scores") or {})), default=0)
    if runs:
        out["runs_per_prompt"] = runs
        if runs >= 2:
            out["live_legacy_windows_noise"] = {
                k: legacy_summary(turns, k) for k in ("before_r1", "after_r1")
            }
            out["resumed_thread_ge_0_92_noise"] = {
                k: sum(1 for t in resumed if isinstance(t.get(k), (int, float)) and t[k] >= OVERRIDE_THRESHOLD)
                for k in ("before_r1", "after_r1")
            }
        out["other_answer_lines"] = other_lines_report(turns, runs)
    return out


async def main_async(args) -> dict[str, Any]:
    from orion.core.bus.async_service import OrionBusAsync

    turns = annotate(load_turns(args.days))
    bus = OrionBusAsync(url=args.bus_url)
    await bus.connect()
    errors = Counter()
    try:
        for i, t in enumerate(turns):
            if i == 0:
                continue
            before, after = prompts_for(t, turns[i - 1])
            t["scores"] = {}
            for run in range(args.runs):
                for key, prompt in (("before", before), ("after", after)):
                    for attempt in range(2):
                        try:
                            s = await classify(bus, prompt, route=args.route, timeout=args.timeout)
                            t["scores"][f"{key}_r{run}"] = {
                                k: s.get(k) for k in ("conversation_boundary_score", "novelty_score", "shift_kind",
                                                      "memory_significance_score", "confidence", "shift_scores")}
                            t[f"{key}_r{run}"] = s.get("conversation_boundary_score")
                            if run == 0:
                                t[key] = s.get("conversation_boundary_score")
                                t[f"{key}_shift"] = s.get("shift_kind")
                            break
                        except Exception as exc:  # noqa: BLE001
                            errors[f"{key}:{type(exc).__name__}"] += 1
            print(f"{i}/{len(turns) - 1} {t['phase']} before={t.get('before')} after={t.get('after')}", flush=True)
    finally:
        await bus.close()
    r = report(turns)
    r.update({"generated_at": datetime.now(timezone.utc).isoformat(), "route": args.route, "days": args.days,
              "errors": dict(errors)})
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--route", default="metacog_background")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--runs", type=int, default=2, help="runs per prompt; run-to-run disagreement is the noise yardstick")
    ap.add_argument("--bus-url", default=DEFAULT_BUS)
    ap.add_argument("--summary-json", type=Path)
    args = ap.parse_args()
    r = asyncio.run(main_async(args))
    text = json.dumps(r, indent=1, default=str)
    if args.summary_json:
        args.summary_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
