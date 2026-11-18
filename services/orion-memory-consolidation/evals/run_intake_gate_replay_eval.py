#!/usr/bin/env python3
"""Replay 30 days of real chat intake windows through the Stage 0A memory gate.

    python services/orion-memory-consolidation/evals/run_intake_gate_replay_eval.py           # replay the fixture
    python services/orion-memory-consolidation/evals/run_intake_gate_replay_eval.py --json    # machine-readable
    python services/orion-memory-consolidation/evals/run_intake_gate_replay_eval.py --refresh # re-capture from live Postgres (read-only)

What it answers: of the chat windows the intake actually judged in the last 30
days, how many would the new gate keep, how many would it drop and why, and are
the real memories (Austin, the offsite, the labs result, the family weekend)
still kept? Label-free: "junk" is whatever the new rule drops; Juniper reads the
kept list if she wants to, she never labels anything.

What the gate sees per window, and where each input comes from on replay:
  * prompt text            -- `memory_consolidation_windows.turn_correlation_ids[].prompt`
  * novelty / shift_kind   -- the same row's `spark_meta.turn_change_appraisal`
  * memory significance    -- the same row's `spark_meta.memory_significance_score`
  * repair signal          -- max `repair_pressure_appraisal_log.level` for the turn's
                              correlation_id, compared against the new floor
                              (`REPAIR_SIGNAL_LEVEL_FLOOR`). The "old" repair share is
                              "any appraisal row exists", which is what
                              `has_repair_signal=repair_bundle is not None` meant.

PRIVACY. This repo is public and the spec says it never holds Juniper's
family or health content. `--refresh` keeps a prompt verbatim only when the
new gate itself classifies it as a greeting/filler or a Hub command (there is
nothing in it), or when it is one of the Austin-day lines the spec already
quotes. Every other prompt is stored as `[redacted prompt: N chars]`; the
labs and family keepers are checked by correlation id, never by text.
Consequence, stated plainly: a redacted prompt is replayed as non-junk by
construction, so this fixture cannot catch a FUTURE rule that over-drops
private content. `--refresh` re-classifies the real text and reports that
case; run it after changing `orion/memory/intake_junk.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.memory.consolidation_gate import consolidation_memory_gate  # noqa: E402
from orion.memory.crystallization.intake_consolidation_window import (  # noqa: E402
    _window_summary_turn,
)
from orion.memory.intake_junk import prompt_junk_reason  # noqa: E402
from orion.substrate.appraisal.contract import REPAIR_SIGNAL_LEVEL_FLOOR  # noqa: E402

FIXTURE = HERE / "fixtures" / "intake_gate_replay_30d.json"
PG_CONTAINER = os.environ.get("ORION_EVAL_PG_CONTAINER", "orion-athena-sql-db")
LOOKBACK_DAYS = 30
MIN_NOVELTY = 0.35  # orion-memory-consolidation settings defaults
MIN_SIGNIFICANCE = 0.40

# Real memories the gate must keep, by the chat turn's correlation id. Labels
# only -- the labs and family text is never written to the repo.
NAMED_KEEPERS: dict[str, str] = {
    "c15a7a3a-7c56-4982-be70-421ac65e5408": "austin",
    "df63a23c-d397-45a4-a60b-b7bb7fbbf871": "offsite",
    "81bb95de-b559-43ab-8906-241d57972c56": "labs",
    "5f519ca1-b467-4e26-8ddc-402edad703a9": "family",
}
# What the named keepers can and cannot prove. Austin and offsite are stored
# verbatim, so the replay really judges their text. Labs and family are
# redacted placeholders, which the rule can never call junk -- in the fixture
# they only prove the window was not dropped for some OTHER reason. The real
# protection for content like theirs is SYNTHETIC_KEEPERS below.
VERBATIM_KEEPERS = frozenset({"austin", "offsite"})

# Synthetic, non-private messages that must always be kept: one per
# over-dropping finding from the review of PR #2457, plus a greeting-prefixed
# real message. Judged on their real text, every run.
SYNTHETIC_KEEPERS: dict[str, str] = {
    "мама умерла сегодня": "non_latin_cyrillic",
    "母が亡くなった": "non_latin_cjk",
    "אמא שלי חולה": "non_latin_hebrew",
    "Mamá está enferma": "accented_latin",
    "I'm not ok": "short_negation",
    "not good": "short_negation",
    "rough day": "short_feeling",
    "Do a journal pass about my labs": "command_plus_content",
    "run a self review on my divorce": "command_plus_content",
    "when is the surgery?": "short_real_question",
    "where is mom?": "short_real_question",
    "who is Sarah?": "short_real_question",
    "hi, my son was diagnosed today": "greeting_prefixed_real",
}

# Austin-day lines the spec (2026-09-30-memory-episode-redesign-design.md)
# already quotes; safe to keep verbatim.
PUBLIC_VERBATIM_IDS = {
    "c15a7a3a-7c56-4982-be70-421ac65e5408",
    "df63a23c-d397-45a4-a60b-b7bb7fbbf871",
}

_CAPTURE_SQL = f"""
WITH w AS (
  SELECT w.memory_window_id, w.closed_at, w.turn_correlation_ids, w.consolidation_status
  FROM memory_consolidation_windows w
  WHERE w.source_platform IS NULL
    AND w.status = 'consolidated'
    AND w.closed_at > now() - interval '{LOOKBACK_DAYS} days'
),
turns AS (
  SELECT w.memory_window_id, t.ord, t.e
  FROM w, jsonb_array_elements(w.turn_correlation_ids) WITH ORDINALITY AS t(e, ord)
)
SELECT coalesce(json_agg(row_to_json(x) ORDER BY x.closed_at), '[]'::json) FROM (
  SELECT w.memory_window_id, w.closed_at, w.consolidation_status,
    (SELECT json_build_object(
        'kind', m.kind, 'status', m.status, 'summary', m.summary,
        'approved_by', m.governance->>'approved_by',
        'gate_reasons', m.provenance->'gate_reasons')
       FROM memory_crystallizations m
      WHERE m.provenance->>'memory_window_id' = w.memory_window_id
      ORDER BY m.created_at LIMIT 1) AS old_row,
    (SELECT json_agg(json_build_object(
        'correlation_id', t.e->>'correlation_id',
        'prompt', coalesce(t.e->>'prompt', ''),
        'novelty_score', t.e->'spark_meta'->'turn_change_appraisal'->'novelty_score',
        'shift_kind', t.e->'spark_meta'->'turn_change_appraisal'->>'shift_kind',
        'memory_significance_score', t.e->'spark_meta'->'memory_significance_score',
        'repair_level', (SELECT max(r.level) FROM repair_pressure_appraisal_log r
                          WHERE r.correlation_id = t.e->>'correlation_id'),
        'appraised', EXISTS (SELECT 1 FROM repair_pressure_appraisal_log r
                              WHERE r.correlation_id = t.e->>'correlation_id'))
      ORDER BY t.ord)
       FROM turns t WHERE t.memory_window_id = w.memory_window_id) AS turns
  FROM w
) x
"""


def _psql_json(sql: str) -> Any:
    out = subprocess.run(
        ["docker", "exec", PG_CONTAINER, "psql", "-U", "postgres", "-d", "conjourney",
         "-v", "ON_ERROR_STOP=1", "-Atc", f"BEGIN READ ONLY; {sql}; ROLLBACK;"],
        capture_output=True, text=True, timeout=120, check=True,
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.strip() and ln not in {"BEGIN", "ROLLBACK"}]
    return json.loads(lines[-1])


def _redact(turn: dict[str, Any]) -> dict[str, Any]:
    prompt = str(turn.get("prompt") or "")
    keep = turn["correlation_id"] in PUBLIC_VERBATIM_IDS or prompt_junk_reason(prompt) is not None
    out = dict(turn)
    out["prompt_chars"] = len(prompt)
    if not keep:
        out["prompt"] = f"[redacted prompt: {len(prompt)} chars]"
        out["redacted"] = True
    else:
        out["redacted"] = False
    return out


def _redact_old_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """The row the OLD intake actually wrote. Its summary is kept only when it
    is junk (the thing this eval counts); otherwise just whether it was."""
    if not row:
        return row
    summary = str(row.get("summary") or "")
    junk = prompt_junk_reason(summary) is not None
    out = {k: v for k, v in row.items() if k != "summary"}
    out["summary_is_junk"] = junk
    out["summary_chars"] = len(summary)
    if junk:
        out["summary"] = summary
    return out


def refresh_fixture() -> dict[str, Any]:
    windows = _psql_json(_CAPTURE_SQL)
    live_report = replay(windows)  # judged on the REAL text, before redaction
    fixture = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback_days": LOOKBACK_DAYS,
        "source": "memory_consolidation_windows (chat, consolidated) + repair_pressure_appraisal_log",
        "windows": [
            {
                **w,
                "old_row": _redact_old_row(w.get("old_row")),
                "turns": [_redact(t) for t in (w.get("turns") or [])],
            }
            for w in windows
        ],
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    redacted_report = replay(fixture["windows"])
    if live_report["kept"] != redacted_report["kept"]:
        print(
            "WARNING: real-text replay and redacted-fixture replay disagree "
            f"(kept {live_report['kept']} vs {redacted_report['kept']}) -- a redacted "
            "prompt the rule now drops; inspect before trusting the fixture.",
            file=sys.stderr,
        )
    return fixture


def _gate_turn(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": t.get("prompt") or "",
        "response": "",  # the gate no longer reads it for junk; never stored
        "spark_meta": {
            "turn_change_appraisal": {
                "novelty_score": t.get("novelty_score"),
                "shift_kind": t.get("shift_kind") or "NONE",
            },
            "memory_significance_score": t.get("memory_significance_score"),
        },
    }


def replay(windows: list[dict[str, Any]]) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    turns_total = turns_appraised = turns_repair_new = 0
    keeper_seen: dict[str, list[str]] = {label: [] for label in NAMED_KEEPERS.values()}
    old_rows = old_junk_rows = old_admitted = 0

    for w in windows:
        turns = w.get("turns") or []
        old_row = w.get("old_row") or {}
        if old_row:
            old_rows += 1
            # Before redaction the summary is present; after, the flag is.
            if old_row.get("summary_is_junk", prompt_junk_reason(str(old_row.get("summary") or "")) is not None):
                old_junk_rows += 1
        if w.get("consolidation_status") == "ok":
            old_admitted += 1
        turns_total += len(turns)
        turns_appraised += sum(1 for t in turns if t.get("appraised"))
        repair = any((t.get("repair_level") or 0.0) >= REPAIR_SIGNAL_LEVEL_FLOOR for t in turns)
        turns_repair_new += sum(
            1 for t in turns if (t.get("repair_level") or 0.0) >= REPAIR_SIGNAL_LEVEL_FLOOR
        )
        gate = consolidation_memory_gate(
            turns=[_gate_turn(t) for t in turns],
            grammar_repair_signal=repair,
            min_novelty=MIN_NOVELTY,
            min_significance=MIN_SIGNIFICANCE,
        )
        # The summary the row would get (same function the intake uses).
        summary_turn = _window_summary_turn(turns) or {}
        summary = str(summary_turn.get("prompt") or "")
        entry = {
            "memory_window_id": w["memory_window_id"],
            "closed_at": w.get("closed_at"),
            "reasons": gate.reasons,
            "summary": summary,
            "summary_chars": summary_turn.get("prompt_chars") or len(summary),
            "summary_is_junk": prompt_junk_reason(summary) is not None,
            "prompts": [t.get("prompt") or "" for t in turns],
        }
        for reason in gate.reasons:
            reasons[f"{gate.action}:{reason}"] += 1
        (kept if gate.action == "propose" else dropped).append(entry)
        for t in turns:
            label = NAMED_KEEPERS.get(t.get("correlation_id") or "")
            if label:
                keeper_seen[label].append(gate.action)

    return {
        "windows": len(windows),
        "old_gate_admitted": old_admitted,
        "old_rows_written": old_rows,
        "old_rows_with_junk_summary": old_junk_rows,
        "kept": len(kept),
        "dropped": len(dropped),
        "reasons": dict(sorted(reasons.items())),
        "repair_signal_share_old": round(turns_appraised / turns_total, 3) if turns_total else 0.0,
        "repair_signal_share_new": round(turns_repair_new / turns_total, 3) if turns_total else 0.0,
        "kept_under_40_chars": sum(1 for k in kept if k["summary_chars"] < 40),
        "kept_with_junk_summary": sum(1 for k in kept if k["summary_is_junk"]),
        # A keeper turn sits in two windows (a closing turn is carried into the
        # next window); it is kept if ANY window holding it is kept.
        "named_keepers": {
            label: ("kept" if "propose" in actions else ("dropped" if actions else "absent"))
            for label, actions in keeper_seen.items()
        },
        "named_keepers_verbatim": sorted(VERBATIM_KEEPERS),
        "synthetic_keepers": {
            prompt: consolidation_memory_gate(
                turns=[_gate_turn({"prompt": prompt})],
                grammar_repair_signal=False,
                min_novelty=MIN_NOVELTY,
                min_significance=MIN_SIGNIFICANCE,
            ).action.replace("propose", "kept").replace("skip", "dropped")
            for prompt in SYNTHETIC_KEEPERS
        },
        "kept_rows": kept,
        "dropped_rows": dropped,
    }


def _print(report: dict[str, Any]) -> None:
    print(
        f"Chat intake windows replayed: {report['windows']}. Old gate admitted "
        f"{report['old_gate_admitted']}; {report['old_rows_written']} rows exist for them, "
        f"{report['old_rows_with_junk_summary']} with a greeting/command as the summary"
    )
    print(f"New gate: kept {report['kept']}, dropped {report['dropped']}")
    print(f"Reasons: {report['reasons']}")
    print(
        f"Repair-signal share of turns: old {report['repair_signal_share_old']:.1%} "
        f"-> new {report['repair_signal_share_new']:.1%} (floor {REPAIR_SIGNAL_LEVEL_FLOOR})"
    )
    print(f"Kept rows whose summary is under 40 chars: {report['kept_under_40_chars']}")
    print(f"Kept rows whose summary is a greeting/command: {report['kept_with_junk_summary']}")
    print(
        f"Named keepers: {report['named_keepers']} "
        f"(only {', '.join(report['named_keepers_verbatim'])} are judged on real text; "
        "labs/family are redacted placeholders)"
    )
    synth = report["synthetic_keepers"]
    print(f"Synthetic keepers kept: {sum(v == 'kept' for v in synth.values())}/{len(synth)}")
    for prompt, verdict in synth.items():
        if verdict != "kept":
            print(f"  DROPPED synthetic keeper: {prompt!r}")
    print("\nDROPPED:")
    for d in report["dropped_rows"]:
        print(f"  {str(d['closed_at'])[:10]}  {d['reasons']}  {' | '.join(d['prompts'])[:110]}")
    print("\nKEPT:")
    for k in report["kept_rows"]:
        print(f"  {str(k['closed_at'])[:10]}  {k['reasons']}  {k['summary'][:110]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="re-capture the fixture from live Postgres (read-only)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.refresh:
        fixture = refresh_fixture()
    else:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report = replay(fixture["windows"])
    report["captured_at"] = fixture.get("captured_at")
    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        print(f"Fixture captured {fixture.get('captured_at')} ({fixture.get('lookback_days')} days)")
        _print(report)
    missing = [k for k, v in report["named_keepers"].items() if v != "kept"]
    missing += [k for k, v in report["synthetic_keepers"].items() if v != "kept"]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
