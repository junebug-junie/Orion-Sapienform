#!/usr/bin/env python3
"""Retire every rev 2 (prose-grouping) personality disposition. DRY RUN by default.

Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md, rev 3 patch R1
("their nodes are retired by a one-time job (backfill protocol, snapshot to
/tmp/personality-rev3-retire/)") and rev 4 build order week 1 ("the old 180 nodes retired, snapshot
first"). This is a PRODUCTION WRITE: Juniper approves the --apply run separately.

    python3 scripts/personality_rev3_retire_prose_nodes.py              # dry run: snapshot + plan only
    python3 scripts/personality_rev3_retire_prose_nodes.py --apply      # write

Target rows: ``personality_disposition`` with ``status <> 'retired'`` and ``origin <> 'choice'`` (rev 2
origins ``recurrence`` and ``seed_self_inquiry``; live 2026-11-13: 27 active + 153 candidate). Choice
nodes (rev 4) are never touched. Evidence rows are never touched (kept for audit).

What --apply does, in ONE transaction (``SELECT ... FOR UPDATE`` on the target rows):
  * ``status = 'retired'``, ``retired_at = now``, ``updated_at = now``;
  * one ``personality_weight_history`` row per node at now (weight, 'retired', evidence_count),
    ``ON CONFLICT DO NOTHING``, so the panel's history shows the retirement.

Run --apply only AFTER the rev 4 orion-durable-runs is deployed. The rev 2 reducer loads prose nodes
at the start of a tick and writes their status from memory, so a rev 2 tick that spans this write
would un-retire them. The rev 4 reducer never loads them. --apply therefore refuses unless the rev 4
reducer has committed a tick: a ``curiosity_sync`` or ``choice`` cursor row exists (only rev 4
writes those). ``--allow-before-rev4`` overrides it (only with PERSONALITY_ENABLED=false).

CLAUDE.md section 14 (backfill protocol). Each run writes to its own timestamped directory
``<--out>/<UTC run time>[-apply]/`` (default --out /tmp/personality-rev3-retire/), so a later dry run
never overwrites the snapshot an apply was made from:
  before.csv        every target row as it was, written BEFORE any write (dry run too)
  progress.log      one line per step: title, percent, ETA, rows done/total, rate, errors, anomalies
  before_after.csv  disposition_id, origin, status before -> after, weight, evidence_count, label
  report.md         verdict, counts, examples, error count, files written
Refuses (exit 2) when the snapshot would exceed 100k rows or 100 MB; --apply exits 3 before the rev 4 reducer ticked.

DSN: ``--dsn`` or ``POSTGRES_URI`` (psycopg). Rollback: before.csv holds every prior status; set them
back with one UPDATE per row (``status``, ``retired_at = NULL``).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_OUT = Path("/tmp/personality-rev3-retire")
MAX_ROWS = 100_000
MAX_BYTES = 100 * 1024 * 1024
COLUMNS = ("disposition_id", "origin", "status", "weight", "evidence_count", "chosen_count", "lane_counts",
           "first_seen_at", "last_reinforced_at", "retired_at", "updated_at", "name", "orion_verdict", "label")
REV4_TICKED_SQL = "SELECT count(*) AS n FROM personality_cursor WHERE source IN ('curiosity_sync', 'choice')"
TARGET_SQL = (f"SELECT {', '.join(COLUMNS)} FROM personality_disposition "
              "WHERE status <> 'retired' AND origin <> 'choice' ORDER BY disposition_id")
RETIRE_SQL = ("UPDATE personality_disposition SET status = 'retired', retired_at = %(now)s, updated_at = %(now)s "
              "WHERE disposition_id = ANY(%(ids)s) AND status <> 'retired' AND origin <> 'choice'")
HISTORY_SQL = ("INSERT INTO personality_weight_history (disposition_id, at, weight, status, evidence_count) "
               "VALUES (%(disposition_id)s, %(at)s, %(weight)s, 'retired', %(evidence_count)s) "
               "ON CONFLICT (disposition_id, at) DO NOTHING")


class Progress:
    """Deterministic progress lines to ``progress.log`` (and stdout)."""

    def __init__(self, path: Path, total: int) -> None:
        self.path, self.total, self.t0 = path, max(0, total), time.monotonic()
        self.errors, self.anomalies = 0, []

    def line(self, title: str, done: int) -> str:
        elapsed = max(1e-6, time.monotonic() - self.t0)
        rate = done / elapsed
        pct = 100.0 if self.total == 0 else 100.0 * done / self.total
        eta = 0.0 if rate <= 0 else max(0.0, (self.total - done) / rate)
        text = (f"{datetime.now(timezone.utc).isoformat()} {title} pct={pct:.1f} eta_s={eta:.1f} "
                f"rows={done}/{self.total} rate={rate:.1f}/s errors={self.errors} "
                f"anomalies={';'.join(self.anomalies) or 'none'}")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(text)
        return text


def _cell(v: Any) -> str:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        import json
        return json.dumps(v, sort_keys=True)
    return "" if v is None else str(v)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([_cell(x) for x in r])
    return path.stat().st_size


def anomalies(rows: Sequence[dict]) -> list[str]:
    """Things worth a human look before --apply (recorded, never a refusal)."""
    out = []
    origins = sorted({r["origin"] for r in rows} - {"recurrence", "seed_self_inquiry"})
    if origins:
        out.append(f"unexpected_origins:{','.join(origins)}")
    named = sum(1 for r in rows if r.get("name"))
    if named:
        out.append(f"named_nodes:{named}")
    return out


def plan(rows: Sequence[dict]) -> list[tuple]:
    """Pure: before_after rows (disposition_id, origin, before, after, weight, evidence_count, label)."""
    return [(r["disposition_id"], r["origin"], r["status"], "retired", r["weight"], r["evidence_count"],
             (r.get("label") or "")[:160]) for r in rows]


def report(out: Path, *, applied: bool, rows: Sequence[dict], retired: int, errors: int, notes: Sequence[str],
           files: Sequence[Path]) -> str:
    by = {}
    for r in rows:
        key = f"{r['origin']}/{r['status']}"
        by[key] = by.get(key, 0) + 1
    verdict = ("APPLIED" if applied and errors == 0 else "FAILED" if applied else "DRY RUN (nothing written)")
    lines = [
        "# personality rev 3/4: retire prose nodes", "",
        f"- verdict: {verdict}",
        f"- target rows (non-retired, origin <> 'choice'): {len(rows)}",
        f"- by origin/status: {', '.join(f'{k}={v}' for k, v in sorted(by.items())) or 'none'}",
        f"- retired by this run: {retired}",
        f"- errors: {errors}",
        f"- anomalies: {'; '.join(notes) or 'none'}",
        f"- needs another pass: {'yes' if applied and retired != len(rows) else 'no' if applied else 'run with --apply'}",
        "", "## before -> after examples", "",
    ]
    for r in sorted(rows, key=lambda r: -float(r["weight"] or 0))[:10]:
        lines.append(f"- `{r['disposition_id']}` {r['origin']} {r['status']} -> retired, "
                     f"weight {float(r['weight'] or 0):.3f}, evidence {r['evidence_count']}: {(r.get('label') or '')[:100]}")
    lines += ["", "## files", ""] + [f"- {p}" for p in files]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def run_dir_for(out: Path, now: datetime, apply: bool) -> Path:
    """One directory per run: a later run never overwrites an earlier snapshot."""
    return out / (now.strftime("%Y%m%dT%H%M%S%fZ") + ("-apply" if apply else "-dry"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dsn", default=os.getenv("POSTGRES_URI", ""))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--allow-before-rev4", action="store_true",
                    help="apply even though the rev 4 reducer has not ticked (only with PERSONALITY_ENABLED=false)")
    args = ap.parse_args(argv)
    if not args.dsn:
        ap.error("--dsn or POSTGRES_URI is required")

    import psycopg
    from psycopg.rows import dict_row

    run_dir = run_dir_for(args.out, datetime.now(timezone.utc), args.apply)
    run_dir.mkdir(parents=True, exist_ok=False)
    args.out = run_dir
    log = args.out / "progress.log"
    with psycopg.connect(args.dsn, row_factory=dict_row) as conn:
        with conn.transaction():
            conn.execute("SET LOCAL lock_timeout = '10s'")
            if args.apply and not args.allow_before_rev4:
                if not conn.execute(REV4_TICKED_SQL).fetchone()["n"]:
                    Progress(log, 0).line("refused: the rev 4 reducer has not ticked yet (deploy it first)", 0)
                    return 3
            lock = " FOR UPDATE" if args.apply else ""
            rows = [dict(r) for r in conn.execute(TARGET_SQL + lock).fetchall()]
            prog = Progress(log, len(rows))
            if len(rows) > MAX_ROWS:
                prog.anomalies.append(f"too_many_rows:{len(rows)}")
                prog.line("refused snapshot over 100k rows", 0)
                return 2
            size = write_csv(args.out / "before.csv", COLUMNS, ([r[c] for c in COLUMNS] for r in rows))
            if size > MAX_BYTES:
                prog.anomalies.append(f"snapshot_bytes:{size}")
                prog.line("refused snapshot over 100MB", 0)
                return 2
            prog.anomalies.extend(anomalies(rows))
            prog.line("snapshot written before.csv", 0)
            write_csv(args.out / "before_after.csv",
                      ("disposition_id", "origin", "status_before", "status_after", "weight", "evidence_count", "label"),
                      plan(rows))
            retired = 0
            if args.apply and rows:
                now = datetime.now(timezone.utc)
                try:
                    cur = conn.execute(RETIRE_SQL, {"now": now, "ids": [r["disposition_id"] for r in rows]})
                    retired = cur.rowcount or 0
                    with conn.cursor() as c:
                        c.executemany(HISTORY_SQL, [{"disposition_id": r["disposition_id"], "at": now,
                                                     "weight": float(r["weight"] or 0.0),
                                                     "evidence_count": int(r["evidence_count"] or 0)} for r in rows])
                except Exception as exc:  # noqa: BLE001 - the transaction rolls back; report says FAILED
                    prog.errors += 1
                    prog.anomalies.append(f"apply_failed:{type(exc).__name__}")
                    prog.line("apply failed, rolled back", 0)
                    report(args.out, applied=True, rows=rows, retired=0, errors=prog.errors, notes=prog.anomalies,
                           files=[log, args.out / "before.csv", args.out / "before_after.csv", args.out / "report.md"])
                    raise
                if retired != len(rows):
                    prog.anomalies.append(f"retired_{retired}_of_{len(rows)}")
                prog.line("retired", retired)
            else:
                prog.line("dry run: plan written, nothing changed", len(rows))
    files = [log, args.out / "before.csv", args.out / "before_after.csv", args.out / "report.md"]
    print(report(args.out, applied=args.apply, rows=rows, retired=retired, errors=prog.errors,
                 notes=prog.anomalies, files=files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
