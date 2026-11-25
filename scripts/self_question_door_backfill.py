#!/usr/bin/env python3
"""One-time door pass over the Orion-written self-questions that predate the door (CLAUDE.md section 14).

Design: docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md ("Columns", "The 88 existing
Orion rows"). Rows written before the door have ``door IS NULL`` and stay pickable. This pass puts them
through the same door as every new question:

1. **Target**: ``curiosity_self_questions`` rows with ``minted_by = 'orion' AND door IS NULL`` (88 on
   2026-11-25). Juniper's rows are never touched.
2. **Snapshot** every target row, in full, to ``/tmp/self-question-door-backfill/snapshot.jsonl`` before
   anything is written. Refuses above 100k rows (it is ~90).
3. **Write** (only with ``--apply``): ``door = 'held'``, ``door_attempts = 0`` on exactly the snapshotted ids
   that still have ``door IS NULL``. That is the only write this script makes.
4. **Judge**: Hub's door loop (``services/orion-hub/scripts/self_question_door.py``) judges held rows, 8 per
   tick every 300 s, with the same judge and model gate as every new question. So the judging is the loop's,
   not this script's: there is one judge path.
5. **Monitor**: polls the snapshotted ids every ``--poll-sec`` and appends a line to
   ``/tmp/self-question-door-backfill/progress.log`` (percent, ETA, rows done/total, rate, judge errors,
   anomalies). Done when no target row is held, or at ``--max-wait-h``.
6. **Report**: ``report.md`` and ``before_after.csv`` in the same directory.

Needs the migration applied (it refuses when the door columns are missing) and Hub deployed with the door on.
If the loop is not running, the held rows wait (not picked, not lost) and the monitor reads 0% done; it
stops at ``--max-wait-h`` and the report says incomplete.

    python3 scripts/self_question_door_backfill.py              # dry run: target + snapshot, no write
    python3 scripts/self_question_door_backfill.py --apply      # snapshot, hold, monitor, report
    python3 scripts/self_question_door_backfill.py --monitor    # resume monitoring an applied pass
    tail -f /tmp/self-question-door-backfill/progress.log

Rollback (puts the rows back as before the pass: door NULL, door columns at their defaults; Hub's own later door
writes to them are lost, focus rows already queued stay): ``--rollback``, one transaction for every snapshotted id.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

JOB = Path("/tmp/self-question-door-backfill")
PSQL = ["docker", "exec", "-i", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney",
        "-v", "ON_ERROR_STOP=1", "-At"]
ROW_LIMIT = 100_000
TARGET_SQL = ("SELECT row_to_json(q) FROM curiosity_self_questions q "
              "WHERE minted_by = 'orion' AND door IS NULL ORDER BY created_at, question_id")
STATE_SQL = ("SELECT json_build_object('question_id', question_id, 'door', door, 'door_reason', door_reason, "
             "'door_attempts', door_attempts, 'door_judge_model', door_judge_model) "
             "FROM curiosity_self_questions WHERE minted_by = 'orion' AND question_id = ANY(:'ids'::text[])")
HAS_DOOR_SQL = ("SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'curiosity_self_questions' AND column_name = 'door_attempts'")


def psql(sql: str, **vars_: str) -> str:
    cmd = list(PSQL)
    for k, v in vars_.items():
        cmd += ["-v", f"{k}={v}"]
    out = subprocess.run(cmd, input=sql + "\n", check=True, capture_output=True, text=True)
    return out.stdout


def rows_json(sql: str, **vars_: str) -> list[dict]:
    return [json.loads(x) for x in psql(sql, **vars_).splitlines() if x.strip().startswith("{")]


def pg_array(ids: list[str]) -> str:
    for i in ids:
        if not all(c.isalnum() or c in "_.:-" for c in i):
            raise ValueError(f"unexpected question id {i!r}")
    return "{" + ",".join(ids) + "}"


def log(line: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with (JOB / "progress.log").open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {line}\n")
    print(line)


def progress_line(states: list[dict], total: int, started: float, now: float) -> tuple[str, int]:
    """One progress line and the number still held. Pure apart from its inputs."""
    doors = Counter(s.get("door") or "null" for s in states)
    held = doors.get("held", 0)
    done = len(states) - held - doors.get("null", 0)  # a row that vanished is an anomaly, not done
    errors = sum(1 for s in states if s.get("door") == "held" and int(s.get("door_attempts") or 0) > 0)
    elapsed = max(1.0, now - started)
    rate = done / elapsed * 3600.0
    eta = "?" if done <= 0 else f"{(total - done) / (done / elapsed) / 60:.0f} min"
    anomalies = []
    if len(states) < total:
        anomalies.append(f"{total - len(states)} target rows no longer found")
    if doors.get("null"):
        anomalies.append(f"{doors['null']} rows back to door NULL")
    pct = 100.0 * done / total if total else 100.0
    line = (f"self-question door backfill {pct:.0f}% done ETA {eta} rows {done}/{total} rate {rate:.1f}/h "
            f"errors {errors} doors {dict(sorted(doors.items()))} anomalies {anomalies or 'none'}")
    return line, held


def write_report(snapshot: list[dict], states: list[dict]) -> None:
    by_id = {s["question_id"]: s for s in states}
    with (JOB / "before_after.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "door_before", "door_after", "attempts", "judge_model", "reason", "text"])
        for r in snapshot:
            s = by_id.get(r["question_id"], {})
            w.writerow([r["question_id"], r.get("door"), s.get("door"), s.get("door_attempts"), s.get("door_judge_model"),
                        (s.get("door_reason") or "")[:200], (r.get("text") or "")[:200]])
    doors = Counter(s.get("door") or "null" for s in states)
    held = doors.get("held", 0)
    verdict = "complete" if not held else f"incomplete: {held} still held"
    examples = [f"- `{r['question_id']}` -> **{by_id.get(r['question_id'], {}).get('door')}**: "
                f"{(r.get('text') or '')[:140]}" for r in snapshot[:8]]
    (JOB / "report.md").write_text("\n".join([
        "# Self-question door backfill", "",
        f"Verdict: {verdict}.", "",
        f"Rows targeted: {len(snapshot)} (minted_by='orion', door NULL before the pass).",
        f"Doors after: {dict(sorted(doors.items()))}.",
        f"Unjudged share: {doors.get('unjudged', 0)}/{len(snapshot)} (acceptance check 5: under 5%).",
        f"Needs another pass: {'yes' if held or doors.get('unjudged', 0) * 20 > len(snapshot) else 'no'}.", "",
        "Before/after examples:", *examples, "",
        f"Files: {JOB / 'snapshot.jsonl'}, {JOB / 'progress.log'}, {JOB / 'before_after.csv'}, {JOB / 'report.md'}", ""]),
        encoding="utf-8")


def load_snapshot() -> list[dict]:
    path = JOB / "snapshot.jsonl"
    if not path.exists():
        raise SystemExit(f"no snapshot at {path}: run without --monitor first")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def monitor(snapshot: list[dict], *, poll_sec: float, max_wait_h: float) -> None:
    ids = pg_array([r["question_id"] for r in snapshot])
    started = time.monotonic()
    states: list[dict] = []
    while True:
        states = rows_json(STATE_SQL, ids=ids)
        line, held = progress_line(states, len(snapshot), started, time.monotonic())
        log(line)
        if not held or time.monotonic() - started > max_wait_h * 3600:
            break
        time.sleep(poll_sec)
    write_report(snapshot, states)
    log(f"report written: {JOB / 'report.md'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="hold the target rows, then monitor (the production write)")
    mode.add_argument("--monitor", action="store_true", help="resume monitoring an applied pass")
    mode.add_argument("--rollback", action="store_true", help="restore the snapshotted door columns")
    ap.add_argument("--poll-sec", type=float, default=120.0)
    ap.add_argument("--max-wait-h", type=float, default=6.0)
    args = ap.parse_args(argv)
    JOB.mkdir(parents=True, exist_ok=True)

    if int((psql(HAS_DOOR_SQL).strip() or "0")) != 1:
        print("refused: curiosity_self_questions has no door columns; apply "
              "services/orion-sql-db/manual_migration_curiosity_self_question_door_v1.sql first", file=sys.stderr)
        return 2
    if args.monitor:
        monitor(load_snapshot(), poll_sec=args.poll_sec, max_wait_h=args.max_wait_h)
        return 0
    if args.rollback:
        # One transaction (psql -1 with ON_ERROR_STOP): all rows restored, or none. Every snapshotted row had
        # door NULL and the door columns at their defaults, so the restore is a constant. Focus rows the door
        # already queued for routed questions are left alone (they are Orion's record of what was offered).
        snap = load_snapshot()
        if any(r.get("door") is not None for r in snap):
            print("refused: the snapshot holds rows that already had a door; restore by hand", file=sys.stderr)
            return 2
        ids = pg_array([r["question_id"] for r in snap])
        out = subprocess.run(PSQL + ["-1", "-v", f"ids={ids}"], input=(
            "UPDATE curiosity_self_questions SET door = NULL, door_reason = NULL, door_judged_at = NULL, "
            "door_judge_model = NULL, door_prompt_sha256 = NULL, door_attempts = 0, door_demoted_by_run = NULL "
            "WHERE minted_by = 'orion' AND question_id = ANY(:'ids'::text[]);\n"), check=True, capture_output=True,
            text=True)
        log(f"rollback: restored the door columns of {len(snap)} snapshotted rows in one transaction ({out.stdout.strip()})")
        return 0

    target = rows_json(TARGET_SQL)
    if len(target) > ROW_LIMIT:
        print(f"refused: {len(target)} rows is over the {ROW_LIMIT} backfill limit; ask Juniper", file=sys.stderr)
        return 2
    if not args.apply:
        # A dry run never touches the apply's snapshot (the rollback reads that one).
        dry = JOB / "snapshot.dryrun.jsonl"
        dry.write_text("".join(json.dumps(r, default=str) + "\n" for r in target), encoding="utf-8")
        print(f"dry run: {len(target)} target rows (minted_by='orion', door NULL) listed in {dry}; nothing written "
              "to the database. Re-run with --apply to snapshot them and hold them for the door loop.")
        return 0
    snap_path = JOB / "snapshot.jsonl"
    if snap_path.exists():
        print(f"refused: {snap_path} exists from an earlier applied pass; use --monitor or --rollback, "
              "or move it aside", file=sys.stderr)
        return 2
    if not target:
        log("nothing to do: no Orion row has door NULL")
        return 0
    snap_path.write_text("".join(json.dumps(r, default=str) + "\n" for r in target), encoding="utf-8")
    log(f"snapshot: {len(target)} target rows (minted_by='orion', door NULL) -> {snap_path}")
    ids = pg_array([r["question_id"] for r in target])
    held = psql("UPDATE curiosity_self_questions SET door = 'held', door_attempts = 0 "
                "WHERE minted_by = 'orion' AND door IS NULL AND question_id = ANY(:'ids'::text[]) "
                "RETURNING question_id", ids=ids)
    n = len([x for x in held.splitlines() if x.strip() and not x.startswith("UPDATE")])
    log(f"held {n} of {len(target)} rows; Hub's door loop judges them (8 per tick, every 300 s)")
    monitor(target, poll_sec=args.poll_sec, max_wait_h=args.max_wait_h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
