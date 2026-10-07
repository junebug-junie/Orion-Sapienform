#!/usr/bin/env python3
"""Report which hand-applied SQL migrations actually reached the live database.

`services/orion-sql-db/*.sql` is applied BY HAND. Nothing records which ones ran, so a migration
that was written, reviewed, merged and never applied looks identical in git to one that is live
(PR #2400: missing column, attention silently degraded; PR #2424: missing table,
orion-hardware-watch crash-looped).

All the deciding logic lives in ``orion/sql_migration_drift.py`` (read its docstring for how it
decides); the same logic runs every 10 minutes inside ``make substrate-ladder-watch``, which cards
a red file to Hub. This is the by-hand view of the same check.

USAGE
    python3 scripts/check_sql_migrations_applied.py                 # last 30 days of migrations
    python3 scripts/check_sql_migrations_applied.py --since-days 0  # every migration ever
    python3 scripts/check_sql_migrations_applied.py --json
    python3 scripts/check_sql_migrations_applied.py --file X.sql    # show one file's verdict
    python3 scripts/check_sql_migrations_applied.py --quiet         # only problems + verify-manually

Exit codes: 0 = everything checkable in the window is applied; 1 = something is missing,
invalid, or a drop did not run; 2 = could not connect / could not read git history (NOT the same
as "drift found", so an infra failure cannot be mistaken for a pass).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion import sql_migration_drift as drift  # noqa: E402


def connect():
    try:
        import psycopg2
    except ImportError:
        print("psycopg2 is not installed; cannot check live state", file=sys.stderr)
        raise SystemExit(2)
    dsn = dict(
        host=os.environ.get("ORION_PG_HOST", "localhost"),
        port=int(os.environ.get("ORION_PG_PORT", "55432")),
        user=os.environ.get("ORION_PG_USER", "postgres"),
        password=os.environ.get("ORION_PG_PASSWORD", os.environ.get("PGPASSWORD", "postgres")),
        dbname=os.environ.get("ORION_PG_DB", "conjourney"),
        connect_timeout=5,
    )
    try:
        conn = psycopg2.connect(**dsn)
    except Exception as exc:
        print(f"could not connect to {dsn['host']}:{dsn['port']}/{dsn['dbname']}: {exc}",
              file=sys.stderr)
        raise SystemExit(2)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only problems and verify-manually files")
    ap.add_argument("--file", help="show a single migration's verdict by filename")
    ap.add_argument("--since-days", type=int, default=drift.DEFAULT_WINDOW_DAYS,
                    help="only alarm on migrations changed in the last N days (0 = all)")
    ap.add_argument("--ref", default="HEAD", help="git ref whose history orders the migrations")
    args = ap.parse_args()

    try:
        times = drift.commit_times(REPO_ROOT, args.ref)
    except Exception as exc:  # noqa: BLE001
        print(f"could not read git history for {args.ref}: {exc}", file=sys.stderr)
        return 2
    files = drift.load_files(REPO_ROOT, times)
    if not files:
        print(f"no .sql files under {drift.MIGRATION_SUBDIR}", file=sys.stderr)
        return 2

    conn = connect()
    try:
        live = drift.load_live_state(conn)
    finally:
        conn.close()
    # The whole corpus is always replayed (a later DROP explains an earlier CREATE); --file only
    # narrows what is printed.
    report = drift.evaluate(files, live, window_days=args.since_days or None)
    if args.file:
        report.files = [f for f in report.files if f.name == Path(args.file).name]
        if not report.files:
            print(f"no migration matching {args.file!r}", file=sys.stderr)
            return 2

    if args.json:
        print(json.dumps([asdict(f) | {"red": f.red, "changed_at": f.changed_at.isoformat()}
                          for f in report.files], indent=2, default=str))
        return 1 if report.red else 0

    for line in drift.iter_human(report, quiet=args.quiet):
        print(line)
    counts: dict[str, int] = {}
    for f in report.files:
        counts[f.status] = counts.get(f.status, 0) + 1
    window = f"last {args.since_days} days" if args.since_days else "all time"
    print()
    print(f"{len(report.files)} migration file(s) replayed: " + ", ".join(
        f"{v} {k.lower()}" for k, v in sorted(counts.items())) + f"; alarm window: {window}")
    old = [f for f in report.files if not f.in_window and f.status in ("MISSING", "INVALID")]
    if old:
        print(f"{len(old)} older file(s) also show drift but are outside the window (marked 'old').")
    vm = report.verify_manually()
    if vm:
        print(f"{len(vm)} data-only migration(s) in the window cannot be verified from the schema.")
    if report.red:
        print(f"\n{len(report.red_files)} migration file(s) are NOT applied to the live database.")
        return 1
    print("Every schema object the migrations in the window declare is present and valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
