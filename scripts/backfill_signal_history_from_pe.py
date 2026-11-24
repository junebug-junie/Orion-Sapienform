#!/usr/bin/env python3
"""Back-fill substrate_node_signal_history from the PE table (PR #3781, Patch 1).

The five non-PE nodes (cabinet/strain/street alarms, room/walkway camera scene change) start
dual-writing to ``substrate_node_signal_history`` when Patch 1 deploys. Later patches switch
readers to that table; a reader that needs 7 days (attention contests, the strain restart seed's
warm-up count) would see a short window on day one. This script copies the last ``--days``
(default 7) of those five nodes' rows from ``substrate_node_prediction_error_history`` into the
new table, tagged with their kind. Values are copied unchanged.

Backfill protocol (CLAUDE.md section 14):
1. Targets: PE-table rows of the five ids, observed_at in the window, with no signal-table row
   for the same (node_id, observed_at).
2. Snapshot every target row to ``<job-dir>/snapshot-<mode>-<time>.jsonl`` FIRST (new file per
   run). These are exactly the keys an apply inserts. Refuses above 100,000 rows (ask Juniper).
3. Progress: one line per batch to ``<job-dir>/progress.log``.
4. ``<job-dir>/report.md`` and ``<job-dir>/before_after.csv`` (per node: PE rows, signal rows
   before and after, value mismatches between the tables on shared keys).

Idempotent: INSERT ... ON CONFLICT (node_id, observed_at) DO NOTHING, so a re-run (or a row the
live dual-write already added) inserts nothing new. Nothing is ever deleted or updated.

Dry run is the DEFAULT: it reads inside a READ ONLY transaction and writes nothing to the
database. ``--apply`` inserts, one commit per batch. Run it AFTER the migration is applied and
substrate-runtime runs the Patch 1 image (so the dual-write covers everything after the window).

    python3 scripts/backfill_signal_history_from_pe.py            # dry run, 7 days
    python3 scripts/backfill_signal_history_from_pe.py --apply    # writes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.substrate.signal_history import (  # noqa: E402
    INSERT_SIGNAL_HISTORY_SQL,
    PE_HISTORY_TABLE,
    SIGNAL_HISTORY_TABLE,
    SIGNAL_NODE_IDS,
    signal_kind_for,
)

JOB = "signal-history-backfill"
DEFAULT_URI = "postgresql://postgres:postgres@localhost:55432/conjourney"
MAX_ROWS = 100_000
BATCH = 1_000

TARGETS_SQL = f"""
SELECT pe.node_id, pe.observed_at, pe.value
FROM {PE_HISTORY_TABLE} pe
LEFT JOIN {SIGNAL_HISTORY_TABLE} sig
  ON sig.node_id = pe.node_id AND sig.observed_at = pe.observed_at
WHERE pe.node_id = ANY(:node_ids) AND pe.observed_at >= :since AND sig.node_id IS NULL
ORDER BY pe.node_id, pe.observed_at
LIMIT :limit
"""

PER_NODE_SQL = f"""
SELECT
    (SELECT count(*) FROM {PE_HISTORY_TABLE} pe
      WHERE pe.node_id = :node_id AND pe.observed_at >= :since) AS pe_rows,
    (SELECT count(*) FROM {SIGNAL_HISTORY_TABLE} sig
      WHERE sig.node_id = :node_id AND sig.observed_at >= :since) AS signal_rows,
    (SELECT count(*) FROM {PE_HISTORY_TABLE} pe
       JOIN {SIGNAL_HISTORY_TABLE} sig
         ON sig.node_id = pe.node_id AND sig.observed_at = pe.observed_at
      WHERE pe.node_id = :node_id AND pe.observed_at >= :since
        AND sig.value IS DISTINCT FROM pe.value) AS value_mismatches,
    (SELECT count(*) FROM {SIGNAL_HISTORY_TABLE} sig
      WHERE sig.node_id = :node_id AND sig.observed_at >= :since
        AND sig.kind IS DISTINCT FROM :kind) AS wrong_kind
"""


def _per_node(conn: Any, *, since: datetime) -> dict[str, dict[str, int]]:
    from sqlalchemy import text

    out: dict[str, dict[str, int]] = {}
    for node_id in SIGNAL_NODE_IDS:
        row = conn.execute(
            text(PER_NODE_SQL),
            {"since": since, "node_id": node_id, "kind": signal_kind_for(node_id)},
        ).fetchone()
        out[node_id] = {
            "pe_rows": int(row[0]), "signal_rows": int(row[1]),
            "value_mismatches": int(row[2]), "wrong_kind": int(row[3]),
        }
    return out


class Progress:
    def __init__(self, path: Path, total: int, title: str) -> None:
        self.path, self.total, self.title = path, total, title
        self.t0 = time.monotonic()

    def line(self, done: int, errors: int, anomalies: int) -> str:
        el = time.monotonic() - self.t0
        rate = done / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("nan")
        pct = 100.0 * done / self.total if self.total else 100.0
        msg = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {self.title} {pct:5.1f}% "
               f"ETA {eta:6.0f}s rows {done}/{self.total} rate {rate:.1f}/s errors {errors} anomalies {anomalies}")
        with self.path.open("a") as fh:
            fh.write(msg + "\n")
        return msg


def run(conn: Any, *, days: float, apply: bool, job_dir: Path, now: datetime,
        commit: Any = None, out: Any = sys.stdout) -> dict[str, Any]:
    """The whole job on one connection. Returns the summary also written to report.md."""
    from sqlalchemy import text

    job_dir.mkdir(parents=True, exist_ok=True)
    since = now - timedelta(days=days)
    before = _per_node(conn, since=since)
    rows = conn.execute(
        text(TARGETS_SQL), {"node_ids": list(SIGNAL_NODE_IDS), "since": since, "limit": MAX_ROWS + 1}
    ).fetchall()
    if len(rows) > MAX_ROWS:
        raise SystemExit(f"{len(rows)}+ target rows > {MAX_ROWS}: stop and ask Juniper (CLAUDE.md section 14)")
    targets: list[dict[str, Any]] = []
    anomalies = 0
    for node_id, observed_at, value in rows:
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        if value is None or not (value == value) or observed_at > now + timedelta(minutes=5):
            anomalies += 1  # NaN / null / future-stamped: snapshotted and reported, never copied
            continue
        targets.append({"node_id": str(node_id), "observed_at": observed_at,
                        "value": float(value), "kind": signal_kind_for(node_id)})
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    mode = "apply" if apply else "dry_run"
    with (job_dir / f"snapshot-{mode}-{stamp}.jsonl").open("x") as fh:
        for t in targets:
            fh.write(json.dumps({**t, "observed_at": t["observed_at"].isoformat()}) + "\n")
    title = f"{JOB} ({'apply' if apply else 'dry run'})"
    progress = Progress(job_dir / "progress.log", len(targets), title)
    print(progress.line(0, 0, anomalies), file=out)
    done = errors = 0
    for i in range(0, len(targets), BATCH):
        batch = targets[i:i + BATCH]
        if apply:
            try:
                conn.execute(text(INSERT_SIGNAL_HISTORY_SQL), batch)
                if commit is not None:
                    commit()
            except Exception as exc:  # noqa: BLE001 -- counted; a failed batch rolls back alone
                errors += len(batch)
                print(f"batch {i // BATCH} failed: {type(exc).__name__}: {exc}"[:300], file=out)
                if hasattr(conn, "rollback"):
                    conn.rollback()
                continue
        done += len(batch)
        print(progress.line(done, errors, anomalies), file=out)
    after = _per_node(conn, since=since) if apply else before
    per_node_targets: dict[str, int] = {}
    for t in targets:
        per_node_targets[t["node_id"]] = per_node_targets.get(t["node_id"], 0) + 1
    with (job_dir / "before_after.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["node_id", "kind", "pe_rows", "signal_rows_before", "targets",
                    "signal_rows_after", "value_mismatches_after", "wrong_kind_after"])
        for node_id in SIGNAL_NODE_IDS:
            b, a = before[node_id], after[node_id]
            w.writerow([node_id, signal_kind_for(node_id), b["pe_rows"], b["signal_rows"],
                        per_node_targets.get(node_id, 0), a["signal_rows"],
                        a["value_mismatches"], a["wrong_kind"]])
    # Proof the design asks for: over the window, every PE row of the five ids has a signal row
    # with the same value and the right kind (signal rows may exceed PE rows only via rows the
    # live dual-write added after the PE prune cut, which does not happen inside the window).
    matched = all(
        after[n]["signal_rows"] >= after[n]["pe_rows"]
        and after[n]["value_mismatches"] == 0 and after[n]["wrong_kind"] == 0
        for n in SIGNAL_NODE_IDS
    )
    if not apply:
        verdict = f"DRY RUN: {len(targets)} row(s) would be inserted; nothing written"
    elif errors:
        verdict = f"INCOMPLETE: {errors} row(s) failed; re-run --apply (idempotent)"
    elif matched:
        verdict = "OK: signal table matches the PE table for all five ids over the window"
    else:
        verdict = "MISMATCH: see before_after.csv"
    summary = {"verdict": verdict, "mode": mode, "since": since.isoformat(), "targets": len(targets),
               "inserted_or_planned": done, "errors": errors, "anomalies": anomalies,
               "matched": matched if apply else None}
    lines = [f"# {JOB} report", "", f"- verdict: {verdict}", f"- mode: {mode}",
             f"- window: observed_at >= {since.isoformat()} ({days:g} days)",
             f"- target rows: {len(targets)}; processed: {done}; errors: {errors}; anomalies skipped: {anomalies}",
             f"- needs another pass: {'yes' if (apply and (errors or not matched)) else 'no' if apply else 'run --apply'}",
             "", "| node | kind | PE rows | signal before | targets | signal after | value mismatches |",
             "|---|---|---|---|---|---|---|"]
    for node_id in SIGNAL_NODE_IDS:
        b, a = before[node_id], after[node_id]
        lines.append(f"| {node_id} | {signal_kind_for(node_id)} | {b['pe_rows']} | {b['signal_rows']} | "
                     f"{per_node_targets.get(node_id, 0)} | {a['signal_rows']} | {a['value_mismatches']} |")
    lines += ["", "Files: " + ", ".join(sorted(p.name for p in job_dir.iterdir()))]
    (job_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(verdict, file=out)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--apply", action="store_true", help="insert rows (default: dry run, writes nothing)")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--job-dir", default=f"/tmp/{JOB}")
    ap.add_argument("--uri", default=os.environ.get("POSTGRES_URI") or DEFAULT_URI)
    args = ap.parse_args(argv)

    from sqlalchemy import create_engine, text

    engine = create_engine(args.uri)
    with engine.connect() as conn:
        if not args.apply:
            conn.execute(text("SET TRANSACTION READ ONLY"))
        summary = run(conn, days=args.days, apply=args.apply, job_dir=Path(args.job_dir),
                      now=datetime.now(timezone.utc), commit=conn.commit, out=sys.stdout)
        if not args.apply:
            conn.rollback()
    print(f"report: {Path(args.job_dir) / 'report.md'}")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
