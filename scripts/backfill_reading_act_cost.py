#!/usr/bin/env python3
"""Back-fill reading acts' GPU lane cost (attention patch 3a, "reading's real cost").

Until patch 3a every read act (``hub.world_pulse_read``) closed with ``gpu_slot_s_by_lane`` NULL:
the fill joined leases on the act's correlation_id, which no lease carries. The writer now fills it
late, once the read ends (orion/grammar/act_ledger.py ``refill_late_costs``), for reads closed in the
last 96 h. This script runs the SAME per-act fill (``late_cost_targets`` / ``late_read_cost`` /
``apply_late_cost``) over the last ``--days`` (default 30), so appetite's p90 lane cost for reading
exists on day one.

Backfill protocol (CLAUDE.md section 14):
1. Targets: closed read acts with no GPU cost and no "late fill" basis line, whose read has ended.
2. Snapshot every target's act_id and current cost_json to ``<job-dir>/snapshot-<mode>-<time>.jsonl``
   FIRST (a new file per run: a later run never overwrites the snapshot an apply needs to be undone).
   Refuses above 100,000 rows (ask Juniper).
3. Progress: one line per batch to ``<job-dir>/progress.log`` (title, percent, ETA, rows, rate,
   errors, anomalies).
4. ``<job-dir>/report.md`` and ``<job-dir>/before_after.csv`` at the end.

Dry run is the DEFAULT: it computes every cost inside a READ ONLY transaction and writes nothing to
the database. ``--apply`` writes, one commit per act; the UPDATE's guard skips an act the live writer
filled meanwhile. ``--restore`` puts back, for every act an ``--apply`` run touched, the cost_json in
the EARLIEST apply snapshot that holds it (only on acts whose cost carries the "late fill" line).

    python3 scripts/backfill_reading_act_cost.py                 # dry run, 30 days
    python3 scripts/backfill_reading_act_cost.py --apply         # after patch 3a is deployed
    python3 scripts/backfill_reading_act_cost.py --restore       # undo from the snapshot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text  # noqa: E402

from orion.grammar.act_ledger import (  # noqa: E402
    LATE_COST_BASIS_PREFIX,
    apply_late_cost,
    late_cost_targets,
    late_read_cost,
)

JOB = "reading-cost-backfill"
DEFAULT_URI = "postgresql://postgres:postgres@localhost:55432/conjourney"
MAX_ROWS = 100_000
# A read holding more than this many slot-seconds is flagged (not skipped): the pool's longest hold
# is 9,000 s, and a read is at most a few retried turns per stage.
ANOMALY_SLOT_S = 6 * 3600.0
PROGRESS_EVERY = 25


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, datetime) else v


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
               f"ETA {eta:6.0f}s rows {done}/{self.total} rate {rate:.2f}/s errors {errors} anomalies {anomalies}")
        with self.path.open("a") as fh:
            fh.write(msg + "\n")
        return msg


def run(conn: Any, *, days: float, apply: bool, job_dir: Path, now: datetime, limit: int = MAX_ROWS + 1,
        commit: Any = None, out: Any = sys.stdout) -> dict[str, Any]:
    """The whole job on one connection. Returns the summary also written to report.md."""
    job_dir.mkdir(parents=True, exist_ok=True)
    targets = late_cost_targets(conn, now=now, since=now - timedelta(days=days), limit=limit)
    if len(targets) > MAX_ROWS:
        raise SystemExit(f"{len(targets)} target rows > {MAX_ROWS}: stop and ask Juniper (CLAUDE.md section 14)")
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with (job_dir / f"snapshot-{'apply' if apply else 'dry_run'}-{stamp}.jsonl").open("x") as fh:
        for t in targets:
            fh.write(json.dumps({"act_id": t["act_id"], "closed_at": _iso(t["closed_at"]),
                                 "cost_json": t["cost_json"]}, default=str) + "\n")
    title = f"{JOB} ({'apply' if apply else 'dry run'})"
    progress = Progress(job_dir / "progress.log", len(targets), title)
    print(progress.line(0, 0, 0), file=out)
    rows: list[dict[str, Any]] = []
    errors = anomalies = written = skipped = 0
    for i, t in enumerate(targets, 1):
        try:
            late = late_read_cost(conn, t, now=now)
            if apply:
                if apply_late_cost(conn, late, now=now):
                    written += 1
                else:
                    skipped += 1
                if commit is not None:
                    commit()
        except Exception as exc:  # noqa: BLE001 -- one bad act is counted, never stops the job
            errors += 1
            rows.append({"act_id": t["act_id"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            # A failed statement aborts the transaction; roll it back (each applied act is already
            # committed) and keep a dry run read-only.
            if hasattr(conn, "rollback"):
                conn.rollback()
                if not apply:
                    conn.execute(text("SET TRANSACTION READ ONLY"))
            continue
        total_s = sum((late.lanes or {}).values())
        anomaly = "no_lease" if not late.lanes else ("over_6h" if total_s > ANOMALY_SLOT_S else "")
        anomalies += bool(anomaly)
        before = t["cost_json"]
        rows.append({"act_id": late.act_id, "closed_at": _iso(t["closed_at"]), "read_end": late.read_end,
                     "turn_ids": late.turn_ids,
                     "before_gpu_slot_s_by_lane": json.dumps((before or {}).get("gpu_slot_s_by_lane")
                                                             if isinstance(before, dict) else None),
                     "after_gpu_slot_s_by_lane": json.dumps(late.lanes), "after_slot_s": round(total_s, 3),
                     "anomaly": anomaly, "error": ""})
        if i % PROGRESS_EVERY == 0 or i == len(targets):
            print(progress.line(i, errors, anomalies), file=out)
    fields = ["act_id", "closed_at", "read_end", "turn_ids", "before_gpu_slot_s_by_lane",
              "after_gpu_slot_s_by_lane", "after_slot_s", "anomaly", "error"]
    with (job_dir / "before_after.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    costed = [r for r in rows if not r.get("error") and r.get("after_slot_s")]
    lanes: dict[str, float] = {}
    for r in costed:
        for k, v in (json.loads(r["after_gpu_slot_s_by_lane"]) or {}).items():
            lanes[k] = lanes.get(k, 0.0) + v
    per = sorted(r["after_slot_s"] for r in costed)
    summary = {
        "mode": "apply" if apply else "dry_run", "now": now.isoformat(), "days": days, "targets": len(targets),
        "gain_cost": len(costed), "no_lease": sum(1 for r in rows if r.get("anomaly") == "no_lease"),
        "errors": errors, "anomalies": anomalies, "written": written, "skipped_filled_meanwhile": skipped,
        "slot_hours": round(sum(per) / 3600.0, 1),
        "slot_hours_by_lane": {k: round(v / 3600.0, 1) for k, v in sorted(lanes.items(), key=lambda kv: -kv[1])},
        "per_read_slot_s": ({"median": round(statistics.median(per)), "p90": round(per[int(0.9 * (len(per) - 1))]),
                             "max": round(per[-1])} if per else None),
        "read_end": dict(sorted(_count(r.get("read_end", "") for r in rows if not r.get("error")).items())),
    }
    _report(job_dir / "report.md", summary, rows)
    return summary


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        key = x.split(" after")[0] if x.startswith("no end") else x
        out[key] = out.get(key, 0) + 1
    return out


def _report(path: Path, s: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    applied = s["mode"] == "apply"
    verdict = ("APPLIED" if applied else "DRY RUN (nothing written)") + (
        f"; {s['errors']} error(s)" if s["errors"] else "")
    examples = [r for r in rows if not r.get("error")][:5]
    lines = [
        f"# {JOB}: {s['mode']}", "",
        f"Verdict: {verdict}.", "",
        f"- Window: last {s['days']:g} days before {s['now']}.",
        f"- Target read acts (closed, no GPU cost yet, read ended): {s['targets']}.",
        f"- Reads that gain a cost: {s['gain_cost']}. No lease found: {s['no_lease']}.",
        f"- GPU slot-hours recovered: {s['slot_hours']} (by lane: {s['slot_hours_by_lane']}).",
        f"- Per read: {s['per_read_slot_s']}.",
        f"- How each read ended: {s['read_end']}.",
        f"- Written: {s['written']}; skipped (filled by the live writer meanwhile): {s['skipped_filled_meanwhile']}.",
        f"- Errors: {s['errors']}. Anomalies (no lease, or > 6 slot-hours): {s['anomalies']}.",
        f"- Needs another pass: {'yes (errors)' if s['errors'] else 'no'}"
        + ("" if applied else "; run with --apply after patch 3a is deployed") + ".", "",
        "## Before / after examples", "",
        "| act_id | before | after | read end |", "|---|---|---|---|",
    ]
    lines += [f"| {r['act_id']} | {r['before_gpu_slot_s_by_lane']} | {r['after_gpu_slot_s_by_lane']} | {r['read_end']} |"
              for r in examples]
    lines += ["", "Files: snapshot-<mode>-<time>.jsonl, progress.log, before_after.csv, report.md (this file).", ""]
    path.write_text("\n".join(lines))


def restore(conn: Any, job_dir: Path, *, commit: Any = None) -> int:
    """Put each act's pre-fill cost_json back (earliest apply snapshot wins), only on acts this fill
    touched (the "late fill" line)."""
    first: dict[str, Any] = {}
    for snap in sorted(job_dir.glob("snapshot-apply-*.jsonl")):
        for line in snap.read_text().splitlines():
            row = json.loads(line)
            first.setdefault(row["act_id"], row)
    n = 0
    for row in first.values():
        res = conn.execute(text(
            "UPDATE grammar_acts SET cost_json = CAST(:c AS jsonb), updated_at = now() "
            "WHERE act_id = :a AND cost_json->'basis'->>'gpu' LIKE :p"),
            {"c": json.dumps(row["cost_json"]) if row["cost_json"] is not None else None,
             "a": row["act_id"], "p": LATE_COST_BASIS_PREFIX + "%"})
        n += res.rowcount or 0
        if commit is not None:
            commit()
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--apply", action="store_true", help="write the costs (default: dry run)")
    ap.add_argument("--restore", action="store_true", help="restore cost_json from the job's snapshot")
    ap.add_argument("--job-dir", type=Path, default=Path("/tmp") / JOB)
    ap.add_argument("--uri", default=os.environ.get("ORION_ANALYSIS_PG_URI", DEFAULT_URI))
    args = ap.parse_args(argv)
    engine = create_engine(args.uri)
    now = datetime.now(timezone.utc)
    with engine.connect() as conn:
        if args.restore:
            print(f"restored {restore(conn, args.job_dir, commit=conn.commit)} act(s) from {args.job_dir}/snapshot-apply-*.jsonl")
            return 0
        if not args.apply:
            conn.execute(text("SET TRANSACTION READ ONLY"))
        summary = run(conn, days=args.days, apply=args.apply, job_dir=args.job_dir, now=now,
                      commit=conn.commit if args.apply else None)
        if not args.apply:
            conn.rollback()
    print(json.dumps(summary, indent=2))
    print(f"report: {args.job_dir}/report.md  before/after: {args.job_dir}/before_after.csv")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
