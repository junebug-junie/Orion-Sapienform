"""Baseline for turn working notes: how often does a chat turn redo the previous turn's steps?

Spec: docs/superpowers/specs/2026-11-05-turn-working-notes-design.md, acceptance checks 1, 2, 7.

For consecutive tool-using chat turns of one Hub session (at most 24 h apart), counts
the steps (tool + command/path, whitespace-normalized) in turn N+1 that turn N already ran:

* repeated_failed: N ran it and it failed.
* repeated_ok:     N ran it and it worked (re-discovery of an answer N already had).

Also reports the share of tool-using turns whose working notes the taint rule would hold
back (acceptance 7: if nearly all are tainted, notes carry almost nothing). That replay
over-counts reading calls: CONTENT_SQL cuts results to 400 chars, so an orion-reading
receipt cannot be checked and counts as held back. The live_* fields are the truth since
the 2026-11-11 deploy: what the harness actually decided, from `working_notes` cockpit rows.

Read-only (ReadOnlyPsql). Usage, from repo root:

    python -m orion.harness.evals.turn_repeat_baseline --days 14 [--json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Iterable

from orion.dig.turn_steps import CONTENT_SQL, Step, parse_rows
from orion.harness.working_notes import outside_content_reasons

PAIR_MAX_GAP = timedelta(hours=24)

SQL = """
WITH turns AS (
  SELECT c.session_id, c.correlation_id, c.created_at
    FROM chat_history_log c
   WHERE c.created_at > now() - interval '{days} days'
     AND c.session_id IS NOT NULL AND c.correlation_id IS NOT NULL
)
SELECT t.session_id, s.correlation_id, s.seq, t.created_at AS turn_at, s.ts,
       {content} AS content
  FROM cockpit_turn_sighting s JOIN turns t USING (correlation_id)
 WHERE s.stage = 'motor_hop'
   AND s.raw->'step'->'raw'->>'type' IN ('assistant', 'user')
 ORDER BY t.created_at, s.seq
"""


LIVE_SQL = """
SELECT (raw->>'written_bytes')::int AS written_bytes, raw->>'tainted_by' AS tainted_by,
       raw->>'carried_forward' AS carried_forward
  FROM cockpit_turn_sighting
 WHERE stage = 'working_notes' AND ts > now() - interval '{days} days'
"""


def measure_live(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Turns that wrote notes, and how many the harness held back (by tool)."""
    wrote = held = carried = 0
    reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        if int(row.get("written_bytes") or 0) <= 0:
            continue
        wrote += 1
        try:
            tainted = json.loads(row.get("tainted_by") or "[]")
        except ValueError:
            tainted = ["(unreadable)"]
        if tainted:
            held += 1
            for name in tainted:
                reasons[str(name)] += 1
        carried += row.get("carried_forward") == "true"
    return {
        "live_turns_with_notes": wrote,
        "live_held_back": held,
        "live_held_back_share": round(held / wrote, 3) if wrote else None,
        "live_carried": carried,
        "live_held_back_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def _key(step: Step) -> str:
    return f"{step.tool}:{' '.join(step.action.split())}"


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text.strip().replace("Z", "+00:00")[:26])


def measure(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    rows = list(rows)
    session_of: dict[str, str] = {}
    content_by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        corr = row.get("correlation_id", "")
        session_of[corr] = row.get("session_id", "")
        try:
            content = json.loads(row.get("content") or "null")
        except ValueError:
            content = None
        if isinstance(content, list):
            content_by_turn[corr].append({"message": {"content": content}})

    steps_by_turn: dict[str, list[Step]] = defaultdict(list)
    for step in parse_rows(rows):
        steps_by_turn[step.correlation_id].append(step)

    turns_by_session: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for corr, steps in steps_by_turn.items():
        turns_by_session[session_of.get(corr, "")].append((_at(steps[0].turn_at), corr))

    pairs = repeated_failed = repeated_ok = turns_with_repeat = 0
    examples: list[dict[str, str]] = []
    for turns in turns_by_session.values():
        turns.sort()
        for (t0, a), (t1, b) in zip(turns, turns[1:]):
            if t1 - t0 > PAIR_MAX_GAP:
                continue
            pairs += 1
            before: dict[str, str] = {}
            for s in steps_by_turn[a]:
                if before.get(_key(s)) != "ok":  # one success in N means N had the answer
                    before[_key(s)] = s.status
            hit = False
            for s in steps_by_turn[b]:
                prior = before.get(_key(s))
                if prior is None:
                    continue
                hit = True
                if prior == "failed":
                    repeated_failed += 1
                else:
                    repeated_ok += 1
                if len(examples) < 10:
                    examples.append({"turn": b, "step": _key(s)[:160], "prior": prior})
            turns_with_repeat += hit

    tool_turns = len(steps_by_turn)
    tainted = {corr: r for corr in steps_by_turn if (r := outside_content_reasons(content_by_turn[corr]))}
    reasons: dict[str, int] = defaultdict(int)
    for r in tainted.values():
        for name in r:
            reasons[name] += 1
    return {
        "tool_using_turns": tool_turns,
        "consecutive_pairs": pairs,
        "pairs_with_a_repeat": turns_with_repeat,
        "repeated_failed": repeated_failed,
        "repeated_ok": repeated_ok,
        "tainted_turns": len(tainted),
        "tainted_share": round(len(tainted) / tool_turns, 3) if tool_turns else None,
        "taint_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    from orion.body_atlas.sources import psql_rows
    from orion.evals.model_replay.sandbox import ReadOnlyPsql

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    psql = ReadOnlyPsql(timeout_sec=300.0)
    days = max(1, ns.days)
    result = measure(psql_rows(psql, SQL.format(days=days, content=CONTENT_SQL)))
    result.update(measure_live(psql_rows(psql, LIVE_SQL.format(days=days))))
    result["days"] = ns.days
    if ns.json:
        print(json.dumps(result, indent=1))
    else:
        for k, v in result.items():
            if k != "examples":
                print(f"{k}: {v}")
        for ex in result["examples"]:
            print(f"  e.g. {ex['prior']:6s} {ex['step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
