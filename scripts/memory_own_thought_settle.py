#!/usr/bin/env python3
"""One-time: stop asking Juniper to confirm Orion's own thoughts (issue #3840).

Moves every ``episode_memory`` row with voice ``orion_thought`` that is still in an asking state
(``pending_confirmation``, ``unconfirmed``, ``not_asked``) to ``own_thought`` and withdraws its open
``memory_confirmation`` card with a recorded reason. It runs the SAME statement the confirmation
tick runs (``orion.memory.episode.confirmation.SETTLE_OWN_THOUGHTS_SQL``), so after deploy the tick
keeps the invariant and this script is only the first, audited pass.

Backfill protocol (AGENTS.md section 14). Writes, all mode 600 under ``/tmp/memory-own-thought/``
(mode 700): ``snapshot.csv`` (every target row and its card, BEFORE the apply; a dry run writes
``snapshot_dry_run.csv``), ``progress.log``,
``before_after.csv`` and ``report.md``. Counts and ids only: no memory text is ever written.

    python scripts/memory_own_thought_settle.py              # dry run: snapshot + counts, no writes
    python scripts/memory_own_thought_settle.py --apply      # snapshot, then move the rows
    python scripts/memory_own_thought_settle.py --rollback   # put back the states in snapshot.csv

Rollback only sticks while the new code is not running: the confirmation tick runs the same statement
every pass and would move the rows straight back (logging a fresh event). Revert the code, or set
MEMORY_CONFIRMATION_LOOP_ENABLED=false on orion-memory-consolidation, before rolling back.

DSN: ``--dsn``, else ``ORION_PAGES_DSN``, else POSTGRES_URI from services/orion-durable-runs/.env
(rewritten to localhost:55432).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.memory.episode.confirmation import (  # noqa: E402
    OWN_THOUGHT_FROM_STATES,
    OWN_THOUGHT_REASON,
    OWN_THOUGHT_STATE,
    OWN_THOUGHT_VOICE,
    SETTLE_OWN_THOUGHTS_SQL,
    SOURCE_KIND,
)

OUT_DIR = Path("/tmp/memory-own-thought")
MAX_ROWS = 100_000  # AGENTS.md section 14: a bigger snapshot needs Juniper first

_STATES_SQL = ", ".join(f"'{st}'" for st in OWN_THOUGHT_FROM_STATES)
# No memory text: ids, labels and states only.
TARGET_SQL = f"""
SELECT m.memory_id::text AS memory_id, m.voice, m.channel, m.stakes, m.status,
       m.confirmation_state, m.confirmation_loop_id, m.updated_at,
       (SELECT string_agg(a.ask_id || ':' || a.status, ',' ORDER BY a.created_at) FROM orion_ask a
         WHERE a.source_kind = '{SOURCE_KIND}' AND a.source_ref = m.confirmation_loop_id) AS cards,
       EXISTS (SELECT 1 FROM attention_loop_outcome o WHERE o.loop_id = m.confirmation_loop_id
               AND NOT EXISTS (SELECT 1 FROM episode_memory_event e WHERE e.outcome_id = o.outcome_id))
         AS answer_waiting
FROM episode_memory m
WHERE m.voice = '{OWN_THOUGHT_VOICE}' AND m.confirmation_state IN ({_STATES_SQL})
ORDER BY m.memory_id
"""
AFTER_SQL = f"""
SELECT m.memory_id::text AS memory_id, m.confirmation_state,
       (SELECT string_agg(a.ask_id || ':' || a.status, ',' ORDER BY a.created_at) FROM orion_ask a
         WHERE a.source_kind = '{SOURCE_KIND}' AND a.source_ref = m.confirmation_loop_id) AS cards
FROM episode_memory m WHERE m.memory_id::text = ANY(%s)
"""
# All orion_thought rows by state (the before/after headline).
STATE_COUNTS_SQL = f"""
SELECT channel, confirmation_state, count(*) AS n FROM episode_memory
WHERE voice = '{OWN_THOUGHT_VOICE}' GROUP BY 1, 2 ORDER BY 1, 2
"""
PENDING_BY_VOICE_SQL = """
SELECT voice, count(*) AS n FROM episode_memory
WHERE confirmation_state = 'pending_confirmation' AND status = 'active' GROUP BY 1 ORDER BY 1
"""
# Rollback: only rows this pass moved (state still own_thought), back to their snapshot state; a
# card withdrawn by this pass reopens only while its own expiry is still ahead.
ROLLBACK_MEMORY_SQL = f"""
UPDATE episode_memory SET confirmation_state = %s, updated_at = now()
WHERE memory_id = %s::uuid AND confirmation_state = '{OWN_THOUGHT_STATE}'
"""
ROLLBACK_CARD_SQL = f"""
UPDATE orion_ask a SET status = 'open'
WHERE a.ask_id = %s AND a.status = 'withdrawn' AND (a.expires_at IS NULL OR a.expires_at > now())
  AND EXISTS (SELECT 1 FROM episode_memory_event e WHERE e.op = '{OWN_THOUGHT_STATE}'
              AND e.reason = '{OWN_THOUGHT_REASON}' AND e.evidence->'withdrawn_ask_ids' ? a.ask_id)
"""

ROLLBACK_EVENT_SQL = f"""
INSERT INTO episode_memory_event (event_id, memory_id, op, actor, episode_id, evidence, reason, created_at)
SELECT gen_random_uuid(), m.memory_id, '{OWN_THOUGHT_STATE}_rolled_back', 'operator.own_thought_settle', m.episode_id,
       jsonb_build_object('restored_state', %s::text), 'operator_rollback', now()
FROM episode_memory m WHERE m.memory_id = %s::uuid
"""

SNAPSHOT_FIELDS = ("memory_id", "voice", "channel", "stakes", "status", "confirmation_state",
                   "confirmation_loop_id", "updated_at", "cards", "answer_waiting")
BEFORE_AFTER_FIELDS = ("memory_id", "channel", "state_before", "state_after", "cards_before", "cards_after")


def default_dsn() -> str:
    if os.environ.get("ORION_PAGES_DSN"):
        return os.environ["ORION_PAGES_DSN"]
    for base in (REPO, Path("/mnt/scripts/Orion-Sapienform")):
        env = base / "services/orion-durable-runs/.env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("POSTGRES_URI="):
                    return line.split("=", 1)[1].strip().replace("@orion-athena-sql-db:5432", "@localhost:55432")
    raise SystemExit("no DSN: pass --dsn or set ORION_PAGES_DSN")


def ensure_out_dir(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    return out


def write_private(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    fields = list(fields)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        os.chmod(tmp, 0o600)
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Progress:
    """One line per step: title, percent, ETA, rows done/total, rate, errors, anomalies."""

    def __init__(self, path: Path, total: int) -> None:
        self.path, self.total, self.t0 = path, max(int(total), 0), time.monotonic()
        if not path.exists():
            write_private(path, "")

    def log(self, title: str, done: int, *, errors: int = 0, anomalies: str = "none") -> str:
        elapsed = max(time.monotonic() - self.t0, 1e-6)
        pct = 100.0 if self.total == 0 else 100.0 * done / self.total
        rate = done / elapsed
        eta = 0.0 if rate <= 0 or done >= self.total else (self.total - done) / rate
        line = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {title} | {pct:.0f}% | "
                f"ETA {eta:.0f}s | rows {done}/{self.total} | {rate:.1f} rows/s | errors {errors} | "
                f"anomalies {anomalies}")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line)
        return line


def before_after_rows(snapshot: list[dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in snapshot:
        a = after.get(str(r["memory_id"]), {})
        out.append({"memory_id": r["memory_id"], "channel": r["channel"],
                    "state_before": r["confirmation_state"], "state_after": a.get("confirmation_state", ""),
                    "cards_before": r.get("cards") or "", "cards_after": a.get("cards") or ""})
    return out


def counts_table(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [f"- {r['channel']} / {r['confirmation_state']}: {r['n']}" for r in rows] or ["- (none)"]


def render_report(*, mode: str, snapshot: list[dict[str, Any]], moved: int, skipped_waiting: int,
                  cards_withdrawn: int, before_counts: list[dict], after_counts: list[dict],
                  pending_before: list[dict], pending_after: list[dict], anomalies: list[str]) -> str:
    by_prior = Counter(f"{r['channel']} / {r['confirmation_state']}" for r in snapshot)
    verdict = ("DRY RUN: nothing written" if mode == "dry-run"
               else ("APPLIED" if not anomalies else "APPLIED WITH ANOMALIES"))
    lines = [
        "# Orion's own thoughts: stop asking Juniper (issue #3840)", "",
        f"Mode: {mode}. Verdict: {verdict}.", "",
        f"Target rows (voice {OWN_THOUGHT_VOICE}, state in {', '.join(OWN_THOUGHT_FROM_STATES)}): {len(snapshot)}",
        *[f"- {k}: {v}" for k, v in sorted(by_prior.items())],
        f"Held back because an answer from Juniper is waiting to be applied: {skipped_waiting}",
        (f"Would move to {OWN_THOUGHT_STATE}: {len(snapshot) - skipped_waiting}" if mode == "dry-run"
         else f"Moved to {OWN_THOUGHT_STATE}: {moved}"),
        f"Open cards {'to withdraw' if mode == 'dry-run' else 'withdrawn'} (reason {OWN_THOUGHT_REASON}): {cards_withdrawn}",
        "", "## orion_thought rows by channel / state, before", *counts_table(before_counts),
        "", "## orion_thought rows by channel / state, after", *counts_table(after_counts),
        "", "## pending_confirmation (active) by voice, before",
        *[f"- {r['voice']}: {r['n']}" for r in pending_before],
        "", "## pending_confirmation (active) by voice, after",
        *[f"- {r['voice']}: {r['n']}" for r in pending_after],
        "", f"Anomalies: {', '.join(anomalies) if anomalies else 'none'}",
        f"Needs another pass: {'yes' if skipped_waiting or anomalies else 'no'} "
        "(the confirmation tick runs the same statement every pass after deploy).",
        "", "Files: snapshot.csv (snapshot_dry_run.csv on a dry run), before_after.csv, progress.log, report.md (this directory).",
        "Rollback: python scripts/memory_own_thought_settle.py --rollback (only sticks with the new code "
        "reverted or MEMORY_CONFIRMATION_LOOP_ENABLED=false: the tick re-runs this statement)", "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    import psycopg
    from psycopg.rows import dict_row

    out = ensure_out_dir(Path(args.out))
    with psycopg.connect(args.dsn or default_dsn(), row_factory=dict_row, autocommit=True) as conn:
        if args.rollback:
            return rollback(conn, out)
        snapshot = conn.execute(TARGET_SQL).fetchall()
        if len(snapshot) > MAX_ROWS:
            raise SystemExit(f"{len(snapshot)} rows > {MAX_ROWS}: stop and ask Juniper (AGENTS.md s14)")
        # The apply snapshot is the rollback source: never overwritten silently. A dry run writes its own.
        snap_path = out / ("snapshot.csv" if args.apply else "snapshot_dry_run.csv")
        if args.apply and snap_path.exists() and not args.overwrite_snapshot:
            raise SystemExit(f"{snap_path} exists: an earlier snapshot would be lost. Pass --overwrite-snapshot.")
        write_csv(snap_path, SNAPSHOT_FIELDS, snapshot)
        prog = Progress(out / "progress.log", len(snapshot))
        prog.log(f"snapshot ({'apply' if args.apply else 'dry-run'})", 0)
        before_counts = conn.execute(STATE_COUNTS_SQL).fetchall()
        pending_before = conn.execute(PENDING_BY_VOICE_SQL).fetchall()
        waiting = sum(1 for r in snapshot if r["answer_waiting"])
        moved, withdrawn, anomalies = 0, 0, []
        if args.apply:
            with conn.transaction():
                result = conn.execute(SETTLE_OWN_THOUGHTS_SQL).fetchall()
            moved = len(result)
            withdrawn = sum(int(r["withdrawn_cards"]) for r in result)
            expected = len(snapshot) - waiting
            if moved != expected:
                anomalies.append(f"moved {moved} != expected {expected}")
            prog.log("settle", moved, anomalies=", ".join(anomalies) or "none")
        else:
            withdrawn = sum(1 for r in snapshot if not r["answer_waiting"] and ":open" in (r.get("cards") or ""))
            prog.log("dry-run (no writes)", 0)
        after = {r["memory_id"]: r for r in conn.execute(
            AFTER_SQL, ([str(r["memory_id"]) for r in snapshot],)).fetchall()}
        write_csv(out / "before_after.csv", BEFORE_AFTER_FIELDS, before_after_rows(snapshot, after))
        report = render_report(
            mode="apply" if args.apply else "dry-run", snapshot=snapshot, moved=moved, skipped_waiting=waiting,
            cards_withdrawn=withdrawn, before_counts=before_counts,
            after_counts=conn.execute(STATE_COUNTS_SQL).fetchall(), pending_before=pending_before,
            pending_after=conn.execute(PENDING_BY_VOICE_SQL).fetchall(), anomalies=anomalies)
        write_private(out / "report.md", report)
        print(report)
        print(f"tail -f {out / 'progress.log'}")
    return 0


def rollback(conn, out: Path) -> int:
    snap = read_csv(out / "snapshot.csv")
    prog = Progress(out / "progress.log", len(snap))
    restored = reopened = 0
    with conn.transaction():
        for r in snap:
            moved_back = conn.execute(ROLLBACK_MEMORY_SQL, (r["confirmation_state"], r["memory_id"])).rowcount
            restored += moved_back
            if not moved_back:
                continue  # the row left own_thought since (superseded, answered): never reopen its card
            conn.execute(ROLLBACK_EVENT_SQL, (r["confirmation_state"], r["memory_id"]))
            for card in filter(None, (r.get("cards") or "").split(",")):
                ask_id, _, status = card.rpartition(":")
                if status == "open":
                    reopened += conn.execute(ROLLBACK_CARD_SQL, (ask_id,)).rowcount
    prog.log(f"rollback (restored {restored}, cards reopened {reopened})", restored)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    mode.add_argument("--rollback", action="store_true", help="restore the states saved in snapshot.csv")
    ap.add_argument("--overwrite-snapshot", action="store_true", help="--apply even if snapshot.csv exists")
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
