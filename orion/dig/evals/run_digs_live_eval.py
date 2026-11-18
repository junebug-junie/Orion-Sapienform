#!/usr/bin/env python3
"""Live eval of `dig digs <date>` (day-dream spec Patch 1 acceptance). Read-only.

For one Denver day (default: yesterday) it runs the command the way Orion does (``cmd_digs``: ReadOnlyPsql in a
READ ONLY transaction, plain file reads of the governor's claude transcripts) and checks:

  E1  it reads in under 3 s;
  E2  every line is a time, a dig command and a ref (check_lines on every call, not only the printed 40);
  E3  no source is UNREAD;
  E4  its chat calls equal a hand-written SQL count of dig-looking tool_use blocks in cockpit_turn_sighting;
  E5  its calls equal an independent scan of every transcript on disk (no mtime or first-line prefilter, every
      line parsed, a plain regex) joined with E4's ids.
      In E4 and E5 the hand count may be larger only by commands the shell check refuses (the dig words sit only
      inside a quoted string or a heredoc: Orion writing a note with printf/cat), and each one must be explained
      by a cruder, separate test (an odd number of quotes, or a `<<`, before every dig word); each is printed;
  E6  every ref is a real row: [chat:] in chat_history_log, [harness_turn_trace:] in harness_turn_trace,
      [durable_run:] in durable_run_workflow;
  E7  at least 90% of calls say where they ran (not `unlinked run`; `chat (no turn id)` says chat);
  E8  it is not empty-shell: at least one call, and at least one with a result line.

It also prints the naive count (motor prompts that merely contain "orion.dig") to show what E4/E5 do not count.

    PYTHONPATH=. python orion/dig/evals/run_digs_live_eval.py [--day YYYY-MM-DD] [--role orion_readonly]
        [--claude-home /root/.claude]

Run it where the transcripts are readable (the harness governor, or a container with its claude-config volume
mounted read-only). ``--role``: default orion_readonly (Orion's own); ``postgres`` still runs READ ONLY.
Exit 0 = every check passed; 1 = a check failed; 2 = the database could not be reached.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.body_atlas.sources import psql_rows  # noqa: E402
from orion.dig import __main__ as dig  # noqa: E402
from orion.dig import digs  # noqa: E402
from orion.evals.model_replay.sandbox import ReadOnlyPsql  # noqa: E402

LIMIT_SEC = 3.0
LINKED_SHARE = 0.9
# The same rule as digs.DIG_RE, written as a Postgres regex by hand (no lookbehind there).
PG_DIG_RE = r"(^|[^[:alnum:]_.])-m[[:space:]]+orion[.]dig($|[^[:alnum:]_.])"
ID_RE = re.compile(r"[A-Za-z0-9:._-]{1,200}")


def _in_list(ids) -> str:
    return ",".join(f"'{i}'" for i in sorted(set(ids)) if ID_RE.fullmatch(i)) or "''"


def hand_sql_ids(runner, day: date) -> dict[str, str]:
    lo, hi = digs.day_bounds(day)
    rows = psql_rows(runner, f"""
SELECT DISTINCT x->>'id' AS id, x->'input'->>'command' AS command
  FROM cockpit_turn_sighting s,
       jsonb_array_elements(CASE jsonb_typeof(s.raw->'step'->'raw'->'message'->'content')
                              WHEN 'array' THEN s.raw->'step'->'raw'->'message'->'content' ELSE '[]'::jsonb END) x
 WHERE s.stage = 'motor_hop'
   AND s.ts >= ('{lo}'::timestamptz AT TIME ZONE 'UTC') AND s.ts < ('{hi}'::timestamptz AT TIME ZONE 'UTC')
   AND x->>'type' = 'tool_use' AND x->'input'->>'command' ~ '{PG_DIG_RE}'""")
    return {r["id"]: r["command"] for r in rows}


def naive_count(runner, day: date) -> int:
    lo, hi = digs.day_bounds(day)
    rows = psql_rows(runner, f"""
SELECT count(*) AS n FROM cockpit_turn_sighting
 WHERE ts >= ('{lo}'::timestamptz AT TIME ZONE 'UTC') AND ts < ('{hi}'::timestamptz AT TIME ZONE 'UTC')
   AND raw::text LIKE '%orion.dig%'""")
    return int(rows[0]["n"]) if rows else 0


DIG_WORDS = re.compile(r"-m\s+orion\.dig")


def explained(command: str) -> bool:
    """Crude and separate from digs.is_dig_command: every dig word sits after a `<<` or inside open quotes."""
    for m in DIG_WORDS.finditer(command):
        before = command[: m.start()].replace("\\'", "").replace('\\"', "")
        if "<<" not in before and before.count("'") % 2 == 0 and before.count('"') % 2 == 0:
            return False
    return True


def same_or_explained(name: str, mine: set[str], hand: dict[str, str]) -> tuple[bool, str, list[str]]:
    refused = sorted(set(hand) - mine)
    bad = [t for t in refused if not explained(hand[t])]
    ok = mine <= set(hand) and not bad
    detail = (f"dig {len(mine)} vs hand {len(hand)}: {len(refused)} refused by the shell check "
              f"({len(refused) - len(bad)} explained); only dig {sorted(mine - set(hand))[:3]}; unexplained {bad[:3]}")
    return ok, detail, [f"  {name} refused {t}: {' '.join(hand[t].split())[:120]!r}" for t in refused]


def hand_transcript_ids(home: Path, day: date) -> dict[str, str]:
    """Every transcript on disk, every line: tool_use id -> command, for dig-looking tool_use blocks in the day."""
    lo, hi = digs.window(day)
    ids: dict[str, str] = {}
    for p in (home / "projects").glob("*/*.jsonl"):
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "orion.dig" not in line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") != "assistant" or e.get("isSidechain"):
                    continue
                ts = datetime.fromisoformat(str(e.get("timestamp", "")).replace("Z", "+00:00"))
                if not lo <= ts < hi:
                    continue
                for b in (e.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") \
                            and re.search(r"(^|[^\w.])-m\s+orion\.dig($|[^\w.])", str((b.get("input") or {}).get("command") or "")):
                        ids[b["id"]] = str(b["input"]["command"])
    return ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", default=(datetime.now(ZoneInfo("America/Denver")).date() - timedelta(days=1)).isoformat())
    ap.add_argument("--role", default="orion_readonly")
    ap.add_argument("--claude-home", default=None)
    args = ap.parse_args(argv)
    day = date.fromisoformat(args.day)
    runner = ReadOnlyPsql(role=args.role, timeout_sec=60.0)
    try:
        psql_rows(runner, "SELECT 1 AS ok")
    except RuntimeError as exc:
        print(f"UNREACHABLE: {exc}")
        return 2
    if args.claude_home:
        home = Path(args.claude_home)
    else:
        from orion.harness.session_resume import claude_home
        home = claude_home()

    with tempfile.TemporaryDirectory() as tmp:
        dig_home = Path(tmp)
        import os
        os.environ["ORION_DIG_HOME"] = str(dig_home)
        t0 = time.monotonic()
        text = dig.cmd_digs(day, dig.Out("eval"), runner=runner, claude_home=home)
        took = time.monotonic() - t0
        got = json.loads(next(dig_home.glob("dig/eval/digs-*.json")).read_text())
    calls = got["calls"]
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")

    print(f"dig digs {day} as {args.role}, transcripts from {home}")
    print("\n".join(text.splitlines()[-3:]))
    check("E1 under 3 s", took < LIMIT_SEC, f"{took:.2f} s")
    rebuilt = [digs.Call(**{k: c[k] for k in digs.Call.__dataclass_fields__}) for c in calls]
    try:
        digs.check_lines(digs.render(rebuilt))
        check("E2 line rule", True, f"{len(rebuilt)} lines, each time + dig command + ref")
    except ValueError as exc:
        check("E2 line rule", False, str(exc))
    check("E3 nothing UNREAD", not got["unread"], json.dumps(got["unread"]))
    sql_cmds = hand_sql_ids(runner, day)
    chat = {c["tool_use_id"] for c in calls if c["source"] == "cockpit"}
    passed, detail, shown = same_or_explained("E4", chat, sql_cmds)
    check("E4 chat = hand SQL", passed, detail)
    print("\n".join(shown))
    hand_cmds = {**hand_transcript_ids(home, day), **sql_cmds}
    mine = {c["tool_use_id"] for c in calls}
    passed, detail, shown = same_or_explained("E5", mine, hand_cmds)
    check("E5 total = independent scan", passed and len(calls) == len(mine), detail)
    print("\n".join(shown[:10]) + (f"\n  ... {len(shown) - 10} more" if len(shown) > 10 else ""))
    refs: dict[str, set[str]] = {}
    for c in calls:
        m = re.fullmatch(r"\[(\w+):([^\]]+)\]", c["ref"])
        if m:
            refs.setdefault(m.group(1), set()).add(m.group(2))
    missing = []
    for kind, table, col in (("chat", "chat_history_log", "id"), ("harness_turn_trace", "harness_turn_trace", "correlation_id"),
                             ("durable_run", "durable_run_workflow", "run_id")):
        want = refs.get(kind, set())
        if want:
            have = {r["k"] for r in psql_rows(runner, f"SELECT {col}::text AS k FROM {table} WHERE {col}::text IN ({_in_list(want)})")}
            missing += [f"{kind}:{x}" for x in want - have]
    check("E6 refs are real rows", not missing,
          f"{ {k: len(v) for k, v in refs.items()} }; missing {missing[:3]}")
    linked = sum(1 for c in calls if c["where"] != "unlinked run")
    share = linked / len(calls) if calls else 0.0
    check("E7 where is known", share >= LINKED_SHARE, f"{linked}/{len(calls)} = {share:.0%}")
    with_result = sum(1 for c in calls if c["first_line"])
    check("E8 not empty", bool(calls) and with_result > 0, f"{len(calls)} calls, {with_result} with a result line")
    by_where: dict[str, int] = {}
    for c in calls:
        by_where[c["where"]] = by_where.get(c["where"], 0) + 1
    print(f"info: by where {dict(sorted(by_where.items(), key=lambda kv: -kv[1]))}")
    print(f"info: naive text match (motor_hop rows whose raw mentions orion.dig) = {naive_count(runner, day)}, "
          f"not counted")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
