#!/usr/bin/env python3
"""Replay goal proposals that orion-sql-writer filed in bus_fallback_log as 'Unknown kind'.

Background
----------
On 2026-11-25, from 08:29Z to about 10:05Z, a stale ``SQL_WRITER_ROUTE_MAP_JSON`` in the local
``services/orion-sql-writer/.env`` had no route for ``memory.field_goals.proposed.v1``. Every goal
the attention runtime published in that window went to ``bus_fallback_log`` with
``error='Unknown kind'`` instead of being stored. The env is fixed; this script stores those goals.

Where they go
-------------
Goals are stored the way sql-writer stores them today: folded into ``field_goal_episode`` (one row
per run of Orion wanting the same target, issue #3924), with the artifact recorded in
``field_goal_episode_artifact``. The earlier per-goal table (``field_goal_provenance_log``, #3929)
lost its writer in #3924 and does not exist in the live database. Same coercion as the live handler
(``app/worker.py``, FieldGoalEpisodeSQL branch): a payload without ``artifact_id`` is refused, a
nested envelope is unwrapped, then ``FieldGoalProvenanceV1.model_validate``; then
``app.field_goal_episode.fold_goal`` in goal-ts order with ``replay=True`` (an old goal folds
into the episodes of its own time, not next to today's newest row).

Safety (CLAUDE.md section 14)
-----------------------------
- Dry run by default: the whole replay runs inside one transaction and is rolled back, so the
  counts (opened / folded / duplicate) are the real ones the apply would produce.
- Idempotent: ``fold_goal`` skips an ``artifact_id`` already in ``field_goal_episode_artifact``.
  That dedup table keeps 48 h (``ARTIFACT_DEDUP_WINDOW``); the script refuses to run on goals
  older than that, where a second run would count them twice.
- bus_fallback_log rows are only read, never changed or deleted.
- ``--apply`` writes one transaction (all or nothing) and a snapshot, progress log, report and
  before/after CSV under ``--job-dir``.

    python services/orion-sql-writer/scripts/replay_field_goals_from_bus_fallback.py --dsn postgresql://...
    python services/orion-sql-writer/scripts/replay_field_goals_from_bus_fallback.py --dsn ... --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parents[1]))

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from orion.schemas.field_goal import FieldGoalProvenanceV1  # noqa: E402

KIND = "memory.field_goals.proposed.v1"
ERROR = "Unknown kind"
JOB_DIR = Path("/tmp/goal-fallback-replay")

SELECT_FALLBACK = text(
    "SELECT id, correlation_id, payload, created_at_ts FROM bus_fallback_log "
    "WHERE kind = :kind AND error = :error ORDER BY id"
)


def coerce(payload: Any) -> FieldGoalProvenanceV1:
    """The live handler's coercion, step for step (app/worker.py, FieldGoalEpisodeSQL branch):
    refuse a payload without artifact_id, unwrap a nested envelope, validate. Raises on refusal."""
    from app.worker import _payload_for_schema_validation

    if isinstance(payload, str):
        payload = json.loads(payload)
    if not (isinstance(payload, dict) and payload.get("artifact_id")):
        raise ValueError("field goal payload has no artifact_id")
    validation_payload, _ = _payload_for_schema_validation(payload, FieldGoalProvenanceV1, kind=KIND)
    return FieldGoalProvenanceV1.model_validate(validation_payload)


def plan(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Fallback rows -> (goals to fold in ts order, refused rows). One goal per artifact_id."""
    goals: list[dict] = []
    refused: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        try:
            goal = coerce(row["payload"])
        except Exception as exc:  # noqa: BLE001 -- refused, reported, never written
            refused.append({"fallback_id": row["id"], "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        if goal.artifact_id in seen:
            refused.append({"fallback_id": row["id"], "error": "duplicate artifact_id in fallback log"})
            continue
        seen.add(goal.artifact_id)
        goals.append({"fallback_id": row["id"], "goal": goal})
    goals.sort(key=lambda g: (g["goal"].ts, g["fallback_id"]))
    return goals, refused


def too_old(goals: list[dict], now: datetime, window) -> list[dict]:
    """Goals older than the artifact dedup window: a second replay could not see them as seen."""
    return [g for g in goals if now - _aware(g["goal"].ts) >= window]


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def replay(sess: Any, goals: list[dict], *, fold, log=None) -> Counter:
    """Fold every goal in order. Caller commits or rolls back."""
    counts: Counter = Counter()
    t0 = time.time()
    for i, g in enumerate(goals, 1):
        counts[fold(sess, g["goal"], envelope_id=None, replay=True)] += 1
        if log and (i % 200 == 0 or i == len(goals)):
            rate = i / max(1e-6, time.time() - t0)
            eta = (len(goals) - i) / max(1e-6, rate)
            log(f"replay field goals | {100 * i / len(goals):.0f}% | ETA {eta:.0f}s | {i}/{len(goals)} | "
                f"{rate:.0f}/s | errors 0 | {dict(counts)}")
    return counts


EPISODE_COLS = ("episode_id", "field_target_id", "first_ts", "last_ts", "repeat_count", "max_salience")


def episodes(sess: Any) -> list[dict]:
    rows = sess.execute(text(f"SELECT {', '.join(EPISODE_COLS)} FROM field_goal_episode ORDER BY first_ts"))
    return [dict(r._mapping) for r in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", default=os.getenv("ORION_PG_DSN"), help="Postgres DSN (conjourney)")
    ap.add_argument("--apply", action="store_true", help="write; omit for a dry run (rolled back)")
    ap.add_argument("--job-dir", type=Path, default=JOB_DIR)
    args = ap.parse_args(argv)
    if not args.dsn:
        print("pass --dsn or set ORION_PG_DSN")
        return 2

    from app.field_goal_episode import ARTIFACT_DEDUP_WINDOW, fold_goal

    job = args.job_dir
    job.mkdir(parents=True, exist_ok=True)
    progress = job / "progress.log"

    def log(line: str) -> None:
        stamped = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {line}"
        print(stamped, flush=True)
        with progress.open("a") as fh:
            fh.write(stamped + "\n")

    engine = create_engine(args.dsn)
    sess = sessionmaker(bind=engine)()
    try:
        rows = [dict(r._mapping) for r in sess.execute(SELECT_FALLBACK, {"kind": KIND, "error": ERROR})]
        goals, refused = plan(rows)
        now = datetime.now(timezone.utc)
        old = too_old(goals, now, ARTIFACT_DEDUP_WINDOW)
        if old:
            log(f"REFUSED: {len(old)} goal(s) older than the {ARTIFACT_DEDUP_WINDOW} dedup window; "
                "a replay could double count them")
            return 2
        before = episodes(sess)
        seen_before = sess.execute(
            text("SELECT count(*) FROM field_goal_episode_artifact WHERE artifact_id = ANY(:ids)"),
            {"ids": [g["goal"].artifact_id for g in goals]}).scalar_one()
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        snap = job / f"snapshot_{'apply' if args.apply else 'dryrun'}_{stamp}.json"
        snap.write_text(json.dumps({
            "fallback_rows": len(rows), "refused": refused, "already_seen_artifacts": seen_before,
            "goals": [{"fallback_id": g["fallback_id"], "artifact_id": g["goal"].artifact_id,
                       "field_target_id": g["goal"].field_target_id, "ts": g["goal"].ts.isoformat()}
                      for g in goals],
            "field_goal_episode_before": before,
        }, indent=1, default=str))
        log(f"{'APPLY' if args.apply else 'DRY RUN'}: fallback rows {len(rows)}, goals {len(goals)}, "
            f"refused {len(refused)}, artifacts already stored {seen_before}, episodes before {len(before)}; "
            f"snapshot {snap}")

        counts = replay(sess, goals, fold=fold_goal, log=log)
        sess.flush()
        after = episodes(sess)
        if args.apply:
            sess.commit()
        else:
            sess.rollback()

        before_ids = {e["episode_id"]: e for e in before}
        changed = [e for e in after if before_ids.get(e["episode_id"]) != e]
        with (job / "before_after.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["episode_id", "field_target_id", "first_ts", "last_ts", "repeat_count_before",
                        "repeat_count_after"])
            for e in changed:
                b = before_ids.get(e["episode_id"]) or {}
                w.writerow([e["episode_id"], e["field_target_id"], e["first_ts"], e["last_ts"],
                            b.get("repeat_count", 0), e["repeat_count"]])
        folded_in = sum(e["repeat_count"] - (before_ids.get(e["episode_id"]) or {}).get("repeat_count", 0)
                        for e in changed)
        verdict = "APPLIED" if args.apply else "DRY RUN (rolled back)"
        report = [
            f"# Goal fallback replay: {verdict}", "",
            f"- bus_fallback_log rows read (kind {KIND}, error '{ERROR}'): {len(rows)} (left in place)",
            f"- refused by the live coercion: {len(refused)}",
            f"- goals replayed in ts order: {len(goals)}",
            f"- outcomes: opened {counts.get('opened', 0)}, folded {counts.get('folded', 0)}, "
            f"duplicate (already stored) {counts.get('duplicate', 0)}",
            f"- field_goal_episode rows: {len(before)} -> {len(after)}; episodes changed {len(changed)}; "
            f"goals counted into them {folded_in}",
            f"- errors: 0", f"- snapshot: {snap}", "",
            "Rollback (not run): delete the episodes listed in before_after.csv whose repeat_count_before is 0, "
            "and their field_goal_episode_artifact rows (episode_id).",
        ]
        (job / "report.md").write_text("\n".join(report) + "\n")
        for line in report[2:8]:
            log(line)
        return 0
    finally:
        sess.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
