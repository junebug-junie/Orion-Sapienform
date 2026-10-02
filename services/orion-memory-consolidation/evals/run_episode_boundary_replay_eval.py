#!/usr/bin/env python3
"""Replay 30 days of Juniper's chat turns through the old and new episode boundaries.

    python services/orion-memory-consolidation/evals/run_episode_boundary_replay_eval.py            # replay the fixture
    python services/orion-memory-consolidation/evals/run_episode_boundary_replay_eval.py --json     # machine-readable
    python services/orion-memory-consolidation/evals/run_episode_boundary_replay_eval.py --refresh  # re-capture (read-only)
    python services/orion-memory-consolidation/evals/run_episode_boundary_replay_eval.py --markdown PATH

Spec: docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md,
Stage 1 acceptance check 1 (episodes per day, turns per episode, close_lag
p50/p95, the Austin day as one episode).

Inputs per turn and where they come from (all read-only):
  * time               -- chat_history_log.created_at (UTC)
  * wall-clock phase   -- recomputed from the gap to Juniper's previous turn with
                          the live classifier (orion.situational.context.
                          classify_conversation_phase, America/Denver). Historical
                          turns carry no stamp (Fix 1 is new), so this is the
                          phase the Hub WOULD have stamped.
  * judge score, two versions, because Fix 2 found they differ:
      - ``first_pass``: the score the live window rule actually saw. It survives
        only on a window's CLOSING entry (the closed window keeps it); every
        other turn's first-pass score was overwritten by the self-comparing
        second pass. A non-closing turn's first-pass score is therefore known
        only to be below 0.85 (it did not close an "unknown phase" window), so
        it is below Rule 3's 0.92 too -- Rule 3 decisions are still exact.
      - ``chat_log``: the score persisted in chat_history_log, i.e. the second,
        self-comparing pass. This is the score the spec's Austin replay used.
  * command turn       -- the response starts with the workflow runtime header.

Rules replayed:
  * legacy_actual      -- the live windows as they were (memory_consolidation_windows).
  * legacy_with_phase  -- the unchanged legacy rule once Fix 1 stamps the phase.
                          Fix 1 changes LIVE window closing through this; reported
                          as a bound because lost first-pass scores could sit in
                          [0.70, 0.85).
  * v2_first_pass      -- Rule 3 with the real judge scores (what will run).
  * v2_chat_log        -- Rule 3 with the artifact scores (the spec's method).

PRIVACY: the fixture stores no prompt or response text, only ids, times,
scores and the command flag.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.situational.context import classify_conversation_phase  # noqa: E402

FIXTURE = HERE / "fixtures" / "episode_boundary_replay_30d.json"
PG_CONTAINER = "orion-athena-sql-db"
TZ = "America/Denver"
LOCAL = ZoneInfo(TZ)

# Same values as services/orion-memory-consolidation/app/settings.py defaults.
SCORE_THRESHOLD = 0.70
LLM_ONLY_THRESHOLD = 0.85
OVERRIDE_THRESHOLD = 0.92
FALLBACK_GAP_SEC = 5400
REORIENT = {"long_gap", "next_day", "stale_thread"}
ACTIVE = {"same_breath", "short_pause"}

# The Austin morning (spec worked example 1), UTC.
AUSTIN_START = datetime(2026, 9, 28, 6, 0, tzinfo=timezone.utc)
AUSTIN_END = datetime(2026, 9, 28, 10, 30, tzinfo=timezone.utc)

_CAPTURE_SQL = r"""
WITH turns AS (
  SELECT correlation_id, created_at,
         (spark_meta->>'conversation_boundary_score')::float AS chat_log_score,
         (ltrim(response) ~ '^Workflow\M') AS is_command
  FROM chat_history_log
  WHERE created_at > now() - interval '30 days'
    AND coalesce(trim(prompt), '') <> '' AND coalesce(trim(response), '') <> ''
    AND client_meta->'external_room' IS NULL
),
closers AS (
  SELECT DISTINCT ON (e->>'correlation_id')
         e->>'correlation_id' AS correlation_id,
         (e->>'conversation_boundary_score')::float AS first_pass_score,
         w.memory_window_id
  FROM memory_consolidation_windows w,
       LATERAL jsonb_array_elements(w.turn_correlation_ids) WITH ORDINALITY a(e, ord)
  WHERE w.source_platform IS NULL AND w.status <> 'open'
    AND w.created_at > now() - interval '32 days'
    AND a.ord = jsonb_array_length(w.turn_correlation_ids)
  ORDER BY e->>'correlation_id', w.created_at
),
windows AS (
  SELECT memory_window_id, created_at, closed_at,
         (SELECT array_agg(e->>'correlation_id' ORDER BY ord)
            FROM jsonb_array_elements(turn_correlation_ids) WITH ORDINALITY a(e, ord)) AS turn_ids
  FROM memory_consolidation_windows
  WHERE source_platform IS NULL AND created_at > now() - interval '30 days'
)
SELECT json_build_object(
  'captured_at', now(),
  'turns', (SELECT coalesce(json_agg(json_build_object(
      'correlation_id', t.correlation_id,
      'at', to_char(t.created_at, 'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"'),
      'chat_log_score', t.chat_log_score,
      'first_pass_score', c.first_pass_score,
      'is_command', t.is_command) ORDER BY t.created_at), '[]'::json)
    FROM turns t LEFT JOIN closers c ON c.correlation_id = t.correlation_id),
  'windows', (SELECT coalesce(json_agg(json_build_object(
      'memory_window_id', memory_window_id,
      'created_at', created_at, 'closed_at', closed_at, 'turn_ids', turn_ids) ORDER BY created_at), '[]'::json)
    FROM windows)
)
"""


def _psql_json(sql: str) -> Any:
    out = subprocess.run(
        ["docker", "exec", PG_CONTAINER, "psql", "-U", "postgres", "-d", "conjourney",
         "-v", "ON_ERROR_STOP=1", "-Atc", f"BEGIN READ ONLY; {sql}; ROLLBACK;"],
        capture_output=True, text=True, timeout=120, check=True,
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.strip() and ln not in {"BEGIN", "ROLLBACK"}]
    return json.loads(lines[-1])


def _ts(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def annotate(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the wall-clock phase each turn would have been stamped with."""
    out = []
    prev: datetime | None = None
    for t in sorted(turns, key=lambda x: x["at"]):
        at = _ts(t["at"])
        clock = classify_conversation_phase(prev, at, TZ)
        out.append({**t, "at_dt": at, "phase": clock["phase_change"], "gap_sec": (at - prev).total_seconds() if prev else None})
        prev = at
    return out


def _rule3(phase: str, score: float | None, gap: float | None) -> bool:
    if phase in REORIENT:
        return True
    if phase == "resumed_thread":
        return score is not None and score >= OVERRIDE_THRESHOLD
    if phase in ACTIVE:
        return False
    return gap is not None and gap >= FALLBACK_GAP_SEC


def _first_pass(t: dict[str, Any], *, lost_as: float) -> float | None:
    if t.get("first_pass_score") is not None:
        return float(t["first_pass_score"])
    return lost_as  # known only to be < 0.85


def split_v2(turns: list[dict[str, Any]], *, score_key: str) -> list[dict[str, Any]]:
    """Rule 3 episodes. The boundary turn opens the next episode."""
    episodes: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for t in turns:
        score = _first_pass(t, lost_as=0.0) if score_key == "first_pass" else t.get("chat_log_score")
        if cur and _rule3(t["phase"], score, t["gap_sec"]):
            episodes.append({"turns": cur, "closed_by": t})
            cur = []
        cur.append(t)
    if cur:
        episodes.append({"turns": cur, "closed_by": None})
    return episodes


def split_legacy_with_phase(turns: list[dict[str, Any]], *, lost_as: float) -> list[dict[str, Any]]:
    """The unchanged legacy rule fed stamped phases. The closing turn ends this
    window AND seeds the next (WindowStore.close_current_window)."""
    windows: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for t in turns:
        cur.append(t)
        score = _first_pass(t, lost_as=lost_as) or 0.0
        phase = t["phase"]
        close = (phase in REORIENT and score >= SCORE_THRESHOLD) or (phase == "unknown" and score >= LLM_ONLY_THRESHOLD)
        if close and len(cur) > 1:
            windows.append({"turns": cur, "closed_by": t})
            cur = [t]
    if cur:
        windows.append({"turns": cur, "closed_by": None})
    return windows


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, int(round(q * (len(vs) - 1)))))
    return vs[idx]


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [g for g in groups if g["closed_by"] is not None]
    sizes = [len(g["turns"]) for g in groups]
    lags = [
        (g["closed_by"]["at_dt"] - g["turns"][-1]["at_dt"]).total_seconds()
        for g in closed
        if g["closed_by"] is not g["turns"][-1]
    ]
    per_day = Counter(g["turns"][0]["at_dt"].astimezone(LOCAL).date().isoformat() for g in groups)
    command_only = sum(1 for g in groups if all(t.get("is_command") for t in g["turns"]))
    return {
        "episodes": len(groups),
        "closed": len(closed),
        "command_only": command_only,
        "days_with_turns": len(per_day),
        "episodes_per_active_day_mean": round(len(groups) / len(per_day), 2) if per_day else None,
        "turns_per_episode_mean": round(statistics.mean(sizes), 2) if sizes else None,
        "turns_per_episode_median": statistics.median(sizes) if sizes else None,
        "close_lag_sec_p50": _pct(lags, 0.5),
        "close_lag_sec_p95": _pct(lags, 0.95),
        "close_reasons": dict(Counter(_reason(g["closed_by"]) for g in closed)),
    }


def _reason(t: dict[str, Any]) -> str:
    # Phase of the closing turn. "unknown" closes by the gap fallback under
    # Rule 3 and by the >=0.85 judge branch under the legacy rule.
    return t["phase"]


def _austin(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for g in groups:
        ts = [t["at_dt"] for t in g["turns"]]
        if any(AUSTIN_START <= x <= AUSTIN_END for x in ts):
            closer = g["closed_by"]
            out.append(
                {
                    "start_utc": min(ts).strftime("%m-%d %H:%M"),
                    "end_utc": max(ts).strftime("%m-%d %H:%M"),
                    "start_mdt": min(ts).astimezone(LOCAL).strftime("%m-%d %H:%M"),
                    "end_mdt": max(ts).astimezone(LOCAL).strftime("%m-%d %H:%M"),
                    "turns": len(ts),
                    "closed_by_utc": closer["at_dt"].strftime("%m-%d %H:%M") if closer else None,
                    "closed_by_phase": closer["phase"] if closer else None,
                    "closed_by_first_pass": closer.get("first_pass_score") if closer else None,
                    "closed_by_chat_log": closer.get("chat_log_score") if closer else None,
                    "close_lag_sec": round((closer["at_dt"] - max(ts)).total_seconds()) if closer and closer["at_dt"] > max(ts) else None,
                }
            )
    return out


def replay(fixture: dict[str, Any]) -> dict[str, Any]:
    turns = annotate(fixture["turns"])
    legacy_actual = [w for w in fixture["windows"] if w.get("turn_ids")]
    actual_days = Counter(_ts(w["created_at"]).astimezone(LOCAL).date().isoformat() for w in legacy_actual)
    actual_sizes = [len(w["turn_ids"]) for w in legacy_actual]
    v2_fp = split_v2(turns, score_key="first_pass")
    v2_cl = split_v2(turns, score_key="chat_log")
    lw_lo = split_legacy_with_phase(turns, lost_as=0.0)
    lw_hi = split_legacy_with_phase(turns, lost_as=0.849)
    # Fix 2 evidence: closing turns whose chat-log score differs from the score that closed them.
    closers = [t for t in turns if t.get("first_pass_score") is not None and t.get("chat_log_score") is not None]
    mismatched = [t for t in closers if abs(float(t["first_pass_score"]) - float(t["chat_log_score"])) > 1e-9]
    resumed = [t for t in turns if t["phase"] == "resumed_thread"]
    return {
        "captured_at": fixture.get("captured_at"),
        "turns": len(turns),
        "phase_histogram": dict(Counter(t["phase"] for t in turns)),
        "fix2": {
            "closing_turns_with_both_scores": len(closers),
            "closing_turns_score_mismatch": len(mismatched),
            "first_pass_mean": round(statistics.mean(float(t["first_pass_score"]) for t in closers), 3) if closers else None,
            "chat_log_mean_same_turns": round(statistics.mean(float(t["chat_log_score"]) for t in closers), 3) if closers else None,
        },
        "resumed_thread_turns": len(resumed),
        "resumed_thread_split_first_pass": sum(1 for t in resumed if _rule3(t["phase"], _first_pass(t, lost_as=0.0), t["gap_sec"])),
        "resumed_thread_split_chat_log": sum(1 for t in resumed if _rule3(t["phase"], t.get("chat_log_score"), t["gap_sec"])),
        "legacy_actual": {
            "windows": len(legacy_actual),
            "windows_per_active_day_mean": round(len(legacy_actual) / len(actual_days), 2) if actual_days else None,
            "turns_per_window_mean": round(statistics.mean(actual_sizes), 2) if actual_sizes else None,
        },
        "legacy_with_phase_lost_low": summarize(lw_lo),
        "legacy_with_phase_lost_high": summarize(lw_hi),
        "v2_first_pass": summarize(v2_fp),
        "v2_chat_log": summarize(v2_cl),
        "austin": {
            "legacy_actual_windows": sum(
                1
                for w in legacy_actual
                if any(AUSTIN_START <= _ts(t["at"]) <= AUSTIN_END for t in fixture["turns"] if t["correlation_id"] in set(w["turn_ids"]))
            ),
            "v2_first_pass": _austin(v2_fp),
            "v2_chat_log": _austin(v2_cl),
        },
    }


def _fmt_lag(sec: float | None) -> str:
    if sec is None:
        return "n/a"
    return f"{sec / 3600:.1f} h" if sec >= 3600 else f"{sec / 60:.0f} min"


def to_markdown(r: dict[str, Any]) -> str:
    lines = [
        "# Episode boundary replay (30 days)",
        "",
        f"Captured {r['captured_at']} (read-only). {r['turns']} of Juniper's direct chat turns. "
        "Times are UTC unless marked MDT.",
        "",
        "## Fix 2 evidence",
        "",
        f"- Closing turns with both scores: {r['fix2']['closing_turns_with_both_scores']}; "
        f"score that closed the window differs from chat_history_log: {r['fix2']['closing_turns_score_mismatch']}.",
        f"- Mean score that closed a window: {r['fix2']['first_pass_mean']}; mean persisted score for the same turns: "
        f"{r['fix2']['chat_log_mean_same_turns']}.",
        "",
        "## Rules compared",
        "",
        "| rule | episodes | per active day | turns/episode (mean, median) | close lag p50 / p95 | close reasons |",
        "|---|---|---|---|---|---|",
    ]
    la = r["legacy_actual"]
    lines.append(f"| legacy, as it ran | {la['windows']} | {la['windows_per_active_day_mean']} | {la['turns_per_window_mean']} | n/a | n/a |")
    for key, label in [
        ("legacy_with_phase_lost_low", "legacy + Fix 1 phase (lost scores = 0)"),
        ("legacy_with_phase_lost_high", "legacy + Fix 1 phase (lost scores = 0.849)"),
        ("v2_first_pass", "Rule 3, real judge scores"),
        ("v2_chat_log", "Rule 3, artifact scores (spec's method)"),
    ]:
        s = r[key]
        lines.append(
            f"| {label} | {s['episodes']} | {s['episodes_per_active_day_mean']} | "
            f"{s['turns_per_episode_mean']}, {s['turns_per_episode_median']} | "
            f"{_fmt_lag(s['close_lag_sec_p50'])} / {_fmt_lag(s['close_lag_sec_p95'])} | {s['close_reasons']} |"
        )
    lines += [
        "",
        f"Phase histogram: {r['phase_histogram']}.",
        f"resumed_thread turns: {r['resumed_thread_turns']}; split by Rule 3 with real scores: "
        f"{r['resumed_thread_split_first_pass']}; with artifact scores: {r['resumed_thread_split_chat_log']}.",
        "",
        "## The Austin morning (2026-09-28)",
        "",
        f"Live windows that morning: {r['austin']['legacy_actual_windows']}.",
        "",
    ]
    for key, label in [("v2_first_pass", "Rule 3, real judge scores"), ("v2_chat_log", "Rule 3, artifact scores")]:
        lines.append(f"**{label}:**")
        lines.append("")
        for e in r["austin"][key]:
            lines.append(
                f"- {e['start_utc']}-{e['end_utc']} UTC ({e['start_mdt']}-{e['end_mdt']} MDT), {e['turns']} turns; "
                f"closed by the {e['closed_by_utc']} turn ({e['closed_by_phase']}, judge {e['closed_by_first_pass']}, "
                f"persisted {e['closed_by_chat_log']}), lag {_fmt_lag(e['close_lag_sec'])}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    if args.refresh:
        data = _psql_json(_CAPTURE_SQL)
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report = replay(fixture)
    if args.markdown:
        args.markdown.write_text(to_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(to_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
