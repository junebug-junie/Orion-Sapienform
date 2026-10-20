#!/usr/bin/env python3
"""One-time backfill: fill late individual links in walkway story evidence (2026-10-20).

A story freezes each crop's ``individual_id`` in ``vision_motion_story.evidence``
when it closes, often seconds before the individuals reducer assigns the crops
(live: 101 of 147 person stories over 5 days said "no individual"). From this
patch on, orion-sql-writer fills those nulls every tick for stories that ended in
the last 6 h (``app.vision_motion_loop.relink_stories``). This script does the
same for older stories, while their crop rows still exist
(``VISION_CROP_RETENTION_DAYS``, 7): after that the link cannot be recovered.

Additive and null-only: an evidence entry that already has an id is never
changed, no entry is added, removed or reordered, and each row is only updated
if its evidence is still exactly what was snapshotted.

Dry run by default (reads only). ``--apply`` first snapshots every candidate row
to ``<job-dir>/snapshot.jsonl`` (CLAUDE.md section 14), then writes, logging to
``<job-dir>/progress.log`` and finishing with ``report.md`` and
``before_after.csv``. ``--restore <job-dir>`` puts the snapshotted evidence back
on rows this script changed (only where the evidence is still what it wrote).

    python services/orion-sql-writer/scripts/backfill_story_evidence_links.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney
    python services/orion-sql-writer/scripts/backfill_story_evidence_links.py --dsn ... --apply
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

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parents[1]))

from app.vision_motion_loop import (  # noqa: E402
    RELINK_CROPS_SQL, RELINK_UPDATE_SQL, fill_evidence_links, unlinked_crop_ids,
)

MAX_SNAPSHOT_ROWS = 100_000
CANDIDATES_SQL = (
    "SELECT story_id, kind, ended_at, evidence FROM vision_motion_story WHERE ended_at >= :since "
    "AND jsonb_typeof(evidence) = 'array' "
    "AND jsonb_path_exists(evidence, '$[*] ? (!exists(@.individual_id) || @.individual_id == null)') "
    "ORDER BY ended_at")


def _ev(raw):
    return json.loads(raw) if isinstance(raw, str) else raw


def plan(conn, since: datetime) -> list[dict]:
    """Rows that would change: story_id, kind, ended_at, old, new, filled."""
    from sqlalchemy import text

    rows = conn.execute(text(CANDIDATES_SQL), {"since": since}).fetchall()
    if len(rows) > MAX_SNAPSHOT_ROWS:
        raise SystemExit(f"{len(rows)} candidate rows > {MAX_SNAPSHOT_ROWS}: stop and ask Juniper (CLAUDE.md 14)")
    ids = sorted({cid for r in rows for cid in unlinked_crop_ids(_ev(r[3]))})
    links = {}
    for i in range(0, len(ids), 5000):
        links.update({r[0]: r[1] for r in conn.execute(text(RELINK_CROPS_SQL), {"ids": ids[i:i + 5000]})})
    out = []
    for sid, kind, ended_at, raw in rows:
        old = _ev(raw)
        new, n = fill_evidence_links(old, links)
        if n:
            out.append({"story_id": sid, "kind": kind, "ended_at": ended_at.isoformat(), "old": old, "new": new,
                        "filled": n})
    return out


def _log(job: Path, line: str) -> None:
    print(line)
    with (job / "progress.log").open("a") as f:
        f.write(line + "\n")


def apply(engine, changes: list[dict], job: Path) -> tuple[int, int]:
    from sqlalchemy import text

    with (job / "snapshot.jsonl").open("w") as f:   # before any write
        for c in changes:
            f.write(json.dumps(c) + "\n")
    done = errors = attempted = 0
    start = time.monotonic()
    total = len(changes)
    for i in range(0, total, 100):
        batch = changes[i:i + 100]
        try:
            with engine.begin() as conn:
                got = sum(conn.execute(text(RELINK_UPDATE_SQL), {
                    "id": c["story_id"], "new": json.dumps(c["new"]), "old": json.dumps(c["old"])}).rowcount or 0
                    for c in batch)
            done += got
            attempted += len(batch)
        except Exception as exc:  # noqa: BLE001 - recorded, next batch continues
            errors += 1
            _log(job, f"batch {i // 100} failed: {exc}")
        n = min(i + 100, total)
        rate = n / max(time.monotonic() - start, 1e-6)
        eta = (total - n) / rate if rate else 0.0
        _log(job, f"story evidence relink {100 * n / max(total, 1):.0f}% eta={eta:.0f}s rows={n}/{total} "
                  f"rate={rate:.0f}/s updated={done} errors={errors} "
                  f"anomalies={attempted - done} (evidence changed since the snapshot: left alone)")
    return done, errors


def report(job: Path, changes: list[dict], done: int, errors: int, applied: bool) -> None:
    with (job / "before_after.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["story_id", "kind", "ended_at", "filled", "ids_before", "ids_after"])
        for c in changes:
            w.writerow([c["story_id"], c["kind"], c["ended_at"], c["filled"],
                        json.dumps([e.get("individual_id") for e in c["old"]]),
                        json.dumps([e.get("individual_id") for e in c["new"]])])
    kinds: dict[str, int] = {}
    for c in changes:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    verdict = "applied" if applied and not errors else "dry run" if not applied else "applied with errors"
    (job / "report.md").write_text(
        f"# Story evidence relink backfill\n\n- verdict: {verdict}\n- stories to change: {len(changes)} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))})\n"
        f"- evidence entries filled: {sum(c['filled'] for c in changes)}\n- rows updated: {done}\n"
        f"- errors: {errors}\n- another pass needed: {'yes (re-run)' if errors or (applied and done < len(changes)) else 'no'}\n"
        f"- files: {job}/{'snapshot.jsonl, ' if applied and changes else ''}progress.log, before_after.csv, "
        f"report.md\n")


def restore(engine, job: Path) -> tuple[int, int]:
    """(rows restored, rows in the snapshot). A row whose evidence changed after the
    backfill (e.g. the loop filled another null) is left alone and shows as a shortfall."""
    from sqlalchemy import text

    n = 0
    lines = (job / "snapshot.jsonl").read_text().splitlines()
    with engine.begin() as conn:
        for line in lines:
            c = json.loads(line)
            n += conn.execute(text(RELINK_UPDATE_SQL), {
                "id": c["story_id"], "new": json.dumps(c["old"]), "old": json.dumps(c["new"])}).rowcount or 0
    return n, len(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", default=os.getenv("POSTGRES_URI", ""))
    ap.add_argument("--days", type=float, default=7.0, help="stories that ended in this many days (crop retention)")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--job-dir", default=f"/tmp/story-evidence-relink-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    ap.add_argument("--restore", metavar="JOB_DIR", help="undo a previous --apply from its snapshot")
    a = ap.parse_args(argv)
    if not a.dsn:
        ap.error("--dsn or POSTGRES_URI is required")
    from sqlalchemy import create_engine

    engine = create_engine(a.dsn, pool_pre_ping=True)
    if a.restore:
        done, total = restore(engine, Path(a.restore))
        print(f"restored {done} of {total} snapshot row(s)"
              + ("" if done == total else " -- the rest changed after the backfill and were left alone"))
        return 0 if done == total else 1
    job = Path(a.job_dir)
    job.mkdir(parents=True, exist_ok=True)
    with engine.connect() as conn:
        changes = plan(conn, datetime.now(timezone.utc) - timedelta(days=a.days))
    _log(job, f"plan: {len(changes)} stories, {sum(c['filled'] for c in changes)} evidence entries to fill "
              f"({'apply' if a.apply else 'dry run'})")
    done = errors = 0
    if a.apply and changes:
        done, errors = apply(engine, changes, job)
    report(job, changes, done, errors, a.apply)
    print(f"report: {job}/report.md")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
