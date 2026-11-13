#!/usr/bin/env python3
"""Back-fill 30 days of camera-ask acts (attention patch 4a, docs/superpowers/specs/2026-11-12-attention-camera-asks-bind-design.md).

From patch 4a every camera ask Orion makes is a ``vision.ask`` act graded by Juniper's answer
(orion/grammar/camera_ask_acts.py). The live loop (app/ask_acts.py) only opens acts for asks of the last
24 h. This script opens one for every older Orion-made camera ask in the window that has none, at the ask's
own ``created_at``, and grades it by the same rule the resolver uses (``resolve_ask_acts``), so the family
is graded on day one instead of after 30 new closes. Juniper's own walkway-story naming cards are excluded.

Additive only: it inserts ``grammar_acts`` rows (plus each close's ``substrate_action_outcomes`` row and the
``vision.ask`` effect-posterior row the act ledger writes on every close). It never changes ``orion_ask``.
An ask that already has an act is skipped, so a second run does nothing.

CLAUDE.md section 14: dry run by default (reads only; writes the plan and a snapshot of the target asks
under ``--job-dir``). ``--apply`` re-snapshots, then writes, logging to ``progress.log`` and finishing with
``report.md`` and ``before_after.csv``. The report prints the rollback SQL (never run by this script).

    python services/orion-sql-writer/scripts/backfill_camera_ask_acts.py --dsn postgresql://postgres:postgres@localhost:55432/conjourney
    python services/orion-sql-writer/scripts/backfill_camera_ask_acts.py --dsn ... --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(SERVICE.parents[1]))

from orion.grammar.camera_ask_acts import ACTOR, SIGNAL, ask_grade, rule_outcome_values  # noqa: E402

DEFAULT_JOB_DIR = Path("/tmp/camera-ask-acts-backfill")
MAX_ROWS = 100_000          # section 14: past this, stop and ask Juniper

STATE_SQL = "SELECT ask_id, status, answer, answer_detail, answered_at, expires_at FROM orion_ask WHERE ask_id = ANY(:ids)"


def _iso(v):
    return v.isoformat() if isinstance(v, datetime) else v


def _plan(conn, *, now: datetime, since: datetime) -> list[dict]:
    from sqlalchemy import text

    from app.ask_acts import asks_needing_acts

    targets = asks_needing_acts(conn, since=since, until=now)
    if not targets:
        return []
    state = {r["ask_id"]: dict(r) for r in conn.execute(text(STATE_SQL), {"ids": [t["ask_id"] for t in targets]}).mappings()}
    for t in targets:
        s = state.get(t["ask_id"]) or {}
        t.update({k: s.get(k) for k in ("answer", "answer_detail", "answered_at")})
        g = ask_grade(s or None, now=now, horizon_at=t["expires_at"])
        t["grade"] = None if g is None else {"value": g.value, "source": g.source, "note": g.note,
                                             "observed_at": _iso(g.observed_at)}
    return targets


def _summary(plan: list[dict]) -> dict:
    graded = [(p["rule"], p["grade"]["value"]) for p in plan if p["grade"] and p["grade"]["source"] == "resolver"]
    return {
        "targets": len(plan),
        "by_rule": dict(Counter(p["rule"] for p in plan)),
        "by_status": dict(Counter(str(p["status"]) for p in plan)),
        "would_grade_1": sum(1 for _, v in graded if v == 1.0),
        "would_grade_0": sum(1 for _, v in graded if v == 0.0),
        "would_close_withdrawn": sum(1 for p in plan if p["grade"] and p["grade"]["source"] == "expired"),
        "stays_open": sum(1 for p in plan if not p["grade"]),
        "graded_closes_by_rule": {r: [sum(1 for rr, v in graded if rr == r and v == 1.0), sum(1 for rr, _ in graded if rr == r)]
                                  for r in sorted({r for r, _ in graded})},
        "expected_outcome_by_rule": {k: round(v, 4) for k, v in rule_outcome_values(graded).items()},
    }


def _write_snapshot(job: Path, plan: list[dict]) -> None:
    job.mkdir(parents=True, exist_ok=True)
    with (job / "snapshot.jsonl").open("w") as f:
        for p in plan:
            f.write(json.dumps({k: _iso(v) for k, v in p.items()}, default=str) + "\n")
    with (job / "plan.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ask_id", "act_id", "rule", "status", "created_at", "expires_at", "grade_value", "grade_source", "note"])
        for p in plan:
            g = p["grade"] or {}
            w.writerow([p["ask_id"], p["act_id"], p["rule"], p["status"], _iso(p["created_at"]), _iso(p["expires_at"]),
                        g.get("value"), g.get("source"), g.get("note")])


def _progress(job: Path, line: str) -> None:
    with (job / "progress.log").open("a") as f:
        f.write(line + "\n")
    print(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--job-dir", type=Path, default=DEFAULT_JOB_DIR)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run, reads only)")
    a = ap.parse_args(argv)

    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=a.days)
    engine = create_engine(a.dsn, pool_pre_ping=True)
    with engine.connect() as conn:
        plan = _plan(conn, now=now, since=since)
    if len(plan) > MAX_ROWS:
        print(f"STOP: {len(plan)} targets > {MAX_ROWS}; ask Juniper (CLAUDE.md section 14)")
        return 2
    summary = _summary(plan)
    _write_snapshot(a.job_dir, plan)
    out = {"mode": "apply" if a.apply else "dry_run", "as_of": now.isoformat(), "since": since.isoformat(), **summary,
           "job_dir": str(a.job_dir)}
    if not a.apply:
        (a.job_dir / "dry_run.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return 0

    from app.ask_acts import open_ask_acts
    from orion.grammar.act_ledger import resolve_ask_acts

    sess = sessionmaker(bind=engine)()
    t0 = time.monotonic()
    try:
        opened = open_ask_acts(sess, now=now, since=since)
        sess.commit()
        _progress(a.job_dir, f"opened acts: {len(opened)}/{len(plan)} 50% elapsed {time.monotonic() - t0:.1f}s errors 0")
        graded, withdrawn = resolve_ask_acts(sess, opened, now=now, commit=sess.commit)
        sess.commit()
        _progress(a.job_dir, f"closed acts: graded {graded} withdrawn {withdrawn} of {len(opened)} 100% "
                              f"elapsed {time.monotonic() - t0:.1f}s")
        rows = sess.execute(text(
            "SELECT act_id, correlation_id, test_before, opened_at, closed_at, outcome_value, outcome_source, outcome_note "
            "FROM grammar_acts WHERE act_id = ANY(:ids) ORDER BY opened_at"), {"ids": opened}).mappings().all()
    finally:
        sess.close()
        engine.dispose()
    with (a.job_dir / "before_after.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ask_id", "act_id", "before", "after_outcome_value", "after_outcome_source", "after_note", "closed_at"])
        for r in rows:
            w.writerow([r["correlation_id"], r["act_id"], "no act", r["outcome_value"], r["outcome_source"],
                        r["outcome_note"], _iso(r["closed_at"])])
    planned_grades = summary["would_grade_1"] + summary["would_grade_0"]
    anomalies = [] if graded == planned_grades and len(opened) == len(plan) else [
        f"planned {len(plan)} opens / {planned_grades} grades, did {len(opened)} / {graded}"]
    ids_sql = ", ".join(f"'{i}'" for i in opened) or "''"
    (a.job_dir / "report.md").write_text(
        f"# Camera-ask acts back-fill ({now.isoformat()})\n\n"
        f"Verdict: {'OK' if not anomalies else 'CHECK ANOMALIES'}\n\n"
        f"- opened {len(opened)} `{ACTOR}` acts (signal `{SIGNAL}`) for asks since {since.isoformat()}\n"
        f"- closed: graded {graded}, withdrawn {withdrawn}, still open {len(opened) - graded - withdrawn}\n"
        f"- by rule: {summary['by_rule']}\n- expected outcome by rule: {summary['expected_outcome_by_rule']}\n"
        f"- errors: 0; anomalies: {anomalies or 'none'}\n- another pass needed: no (re-running skips asks with an act)\n\n"
        "## Rollback (not run by this script)\n\n```sql\n"
        f"DELETE FROM substrate_action_outcomes WHERE dispatch_id IN ({ids_sql});\n"
        f"DELETE FROM grammar_acts WHERE act_id IN ({ids_sql});\n"
        "-- the vision.ask posterior row then over-counts these closes; delete it to restart it cold:\n"
        f"DELETE FROM substrate_action_effect_posterior WHERE target_id = '{ACTOR}' AND signal_id = '{SIGNAL}';\n```\n")
    print(json.dumps({**out, "opened": len(opened), "graded": graded, "withdrawn": withdrawn, "anomalies": anomalies},
                     indent=2))
    return 0 if not anomalies else 1


if __name__ == "__main__":
    raise SystemExit(main())
