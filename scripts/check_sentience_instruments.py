#!/usr/bin/env python3
"""Sentience Striving Program instrument gate.

Re-runs every claim in orion/sentience_striving_program/instruments.yaml against
live repo and database state, and fails when one has drifted from the value the
program recorded.

This exists because the program's narrative record cannot detect its own decay.
On 2026-09-02 a read of README.md found four claims that live data had already
moved past -- including the "only 5 distinct narratives exist" finding that
Objective 7 was closed on, which had since become 16. Prose does not go red.

Usage:
    python scripts/check_sentience_instruments.py            # gate (exit 1 on drift)
    python scripts/check_sentience_instruments.py --report   # always exit 0
    python scripts/check_sentience_instruments.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# scripts/ on sys.path[0] shadows stdlib `platform` via scripts/platform/ and
# breaks pydantic -- same fix as check_metric_lineage.py and
# check_inner_state_registry.py.
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.sentience_striving_program.instruments import (  # noqa: E402
    build_state,
    load_manifest,
)

_MARK = {
    "HOLDS": "ok  ",
    "DRIFTED": "DRIFT",
    "MANUAL": "man ",
    "ERROR": "ERR ",
}


def _open_conn():
    """Read-only Postgres connection, or None with a printed reason.

    Reuses the metric semantic layer's own helper so this script cannot acquire
    write access that layer does not already have.
    """
    try:
        from orion.metrics.liveness import open_readonly_connection

        return open_readonly_connection(), ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _fmt_hours(hours: float | None) -> str:
    if hours is None:
        return "?"
    return f"{hours / 24:.1f}d" if hours >= 48 else f"{hours:.1f}h"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="never exit non-zero")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()

    manifest = load_manifest()
    conn, conn_err = _open_conn()
    try:
        states = build_state(manifest, conn=conn, with_consumers=not args.as_json)
    finally:
        if conn is not None:
            conn.close()

    if args.as_json:
        payload = [
            {
                "id": s.instrument.id,
                "title": s.instrument.title,
                "outcome": s.instrument.outcome,
                "module": s.instrument.module,
                "module_exists": s.module_exists,
                "entrypoint_exists": s.entrypoint_exists,
                "unlock": s.instrument.unlock,
                "last_reviewed": s.instrument.last_reviewed,
                "review_age_days": s.review_age_days,
                "review_stale": s.review_stale,
                "storage_kind": s.instrument.storage.kind,
                "table": s.instrument.storage.table,
                "row_count": s.row_count,
                "history_hours": s.history_hours,
                "retention_hours": s.retention_hours,
                "retention_source": s.retention_source,
                "storage_note": s.storage_note,
                "last_seen": s.last_seen.isoformat() if s.last_seen else None,
                "claims": [
                    {
                        "id": c.claim.id,
                        "question": c.claim.question,
                        "status": c.status,
                        "recorded": c.claim.recorded,
                        "observed": c.observed,
                        "blocks": c.claim.blocks,
                        "detail": c.detail,
                        "note": c.claim.note,
                    }
                    for c in s.claims
                ],
            }
            for s in states
        ]
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print("Sentience Striving Program -- instrument gate")
    if conn is None:
        print(f"  !! no database connection ({conn_err}); SQL claims will ERROR")
    print()

    problems: list[str] = []
    for s in states:
        i = s.instrument
        head = f"  [{i.outcome}] {i.title}"
        print(head)
        print(f"        module   {i.module}", end="")
        if not s.module_exists:
            print("   <-- MISSING")
            problems.append(f"{i.id}: module {i.module} does not exist")
        elif s.entrypoint_exists is False:
            print(f"   <-- entrypoint {i.entrypoint!r} MISSING")
            problems.append(f"{i.id}: entrypoint {i.entrypoint} not found in {i.module}")
        else:
            print()

        if s.row_count is not None:
            span = _fmt_hours(s.history_hours)
            line = f"        data     {s.row_count:,} rows, {span} of history"
            if s.retention_hours:
                line += f", capped at {_fmt_hours(s.retention_hours)} ({s.retention_source})"
            print(line)
        if s.storage_note:
            print(f"        storage  {s.storage_note}")
        if s.consumers:
            print(f"        affects  {len(s.consumers)} real consumers, e.g. {s.consumers[0]}")
        elif s.consumer_note:
            print(f"        affects  ({s.consumer_note})")

        if s.review_stale:
            print(f"        review   STALE -- last read {s.review_age_days}d ago")
            problems.append(
                f"{i.id}: unlock narrative unreviewed for {s.review_age_days}d"
            )

        for c in s.claims:
            blocks = f" [blocks {c.claim.blocks}]" if c.claim.blocks else ""
            print(f"        {_MARK[c.status]} {c.claim.id}{blocks}")
            if c.detail:
                print(f"             {c.detail}")
            if c.status in ("DRIFTED", "ERROR"):
                problems.append(f"{i.id}/{c.claim.id}: {c.status} -- {c.detail}")
        print()

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print()
        print("A DRIFT is not necessarily a regression -- it means live data moved")
        print("past what the program recorded. Re-read the finding, then update")
        print("`recorded`/`recorded_at` in instruments.yaml in the same patch.")
        return 0 if args.report else 1

    print("All claims hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
