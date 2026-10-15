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
    python3 scripts/check_sql_migrations_applied.py --service orion-durable-runs
        # deploy gate: every migration whose header says ORION-MIGRATION-REQUIRED-BY that service
        # must be applied (no recency window). scripts/safe_docker_build.sh runs this before `up`.

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


class ConnectError(RuntimeError):
    pass


def open_connection():
    """Read-only connection, or ConnectError naming why not."""
    try:
        import psycopg2
    except ImportError:
        raise ConnectError("psycopg2 is not installed in this python; cannot check live state")
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
        raise ConnectError(f"could not connect to {dsn['host']}:{dsn['port']}/{dsn['dbname']}: "
                           f"{' '.join(str(exc).split())}")
    conn.set_session(readonly=True, autocommit=True)
    return conn


def connect():
    try:
        return open_connection()
    except ConnectError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2)


def service_gate(service: str, ref: str = "HEAD", connect_fn=None) -> int:
    """Deploy-time gate. One line per problem, each naming the exact apply command.

    Never applies anything. A database it cannot read is UNKNOWN (exit 2), never "applied"."""
    tag = f"migration gate [{service}]"
    files = drift.load_files(REPO_ROOT, {})  # cheap first pass: does anything require this service?
    needed = drift.required_for(files, service)
    if not needed:
        print(f"{tag}: no migration declares ORION-MIGRATION-REQUIRED-BY {service}; nothing to check")
        return 0
    try:
        times = drift.commit_times(REPO_ROOT, ref)
    except Exception as exc:  # noqa: BLE001
        print(f"{tag}: UNKNOWN -- could not read git history for {ref}: {exc}", file=sys.stderr)
        return 2
    files = drift.load_files(REPO_ROOT, times)  # uncommitted files count: a worktree deploys them
    try:
        conn = (connect_fn or open_connection)()
        try:
            live = drift.load_live_state(conn)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"{tag}: UNKNOWN -- cannot read the live database ({exc}), so cannot tell whether "
              f"{len(needed)} required migration(s) are applied; no answer is not 'applied'",
              file=sys.stderr)
        return 2
    report = drift.evaluate(files, live, window_days=None)
    gate = drift.deploy_gate(report, drift.required_for(files, service), service)
    for r in gate.unverifiable:
        print(f"{tag}: note -- {r.name} is data-only/guarded; the schema cannot confirm it ran, "
              "verify manually", file=sys.stderr)
    if gate.blocking:
        for r in gate.blocking:
            print(f"{tag}: MISSING {r.summary()} -- apply: {r.apply_command()}", file=sys.stderr)
        return 1
    print(f"{tag}: all {len(gate.required)} required migration(s) applied: "
          + ", ".join(gate.required))
    return 0


def _required_by_text(text: str) -> tuple[str, bool]:
    """(display string, whether a deploy gate refuses some service on it)."""
    rb = drift.parse_required_by(text)
    if rb is None:
        return "none declared", False
    if rb.services and not drift.is_destructive(text):
        return ", ".join(rb.services), True
    if rb.services:
        return ", ".join(rb.services), False
    return (f"none ({rb.none_reason})" if rb.none_reason else "none declared"), False


def post_pull_report(paths: list[str], ref: str = "HEAD", connect_fn=None) -> list[str]:
    """Read-only lines for scripts/rebuild_services_from_git_diff.sh: one block per changed
    hand-applied migration -- who needs it, live state, and the apply command when one is due.

    Never applies anything. A database it cannot read makes every verdict UNKNOWN, never
    "applied". Rollback and DESTRUCTIVE files get no apply command: a deploy must never run them."""
    lines: list[str] = []
    names = [Path(p).name for p in paths]
    live = None
    why_unknown = ""
    try:
        times = drift.commit_times(REPO_ROOT, ref)
    except Exception as exc:  # noqa: BLE001
        times = {}
        why_unknown = f"could not read git history for {ref}: {exc}"
    files = drift.load_files(REPO_ROOT, times)  # uncommitted files count, as in the deploy gate
    if not why_unknown:
        try:
            conn = (connect_fn or open_connection)()
            try:
                live = drift.load_live_state(conn)
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 -- any failure is UNKNOWN, never a verdict
            why_unknown = " ".join(str(exc).split()) or type(exc).__name__
    # Against an empty live state every file reads MISSING; used only for apply commands.
    empty = drift.LiveState(tables=set(), columns=set(), indexes={}, sequences=set())
    report = drift.evaluate(files, live if live is not None else empty, window_days=None)
    by_name = {f.name: f for f in report.files}
    texts = {f.name: f.text for f in files}
    for rel, name in zip(paths, names):
        path = REPO_ROOT / rel
        text = texts.get(name) or (path.read_text(errors="replace") if path.is_file() else "")
        required, gated = _required_by_text(text)
        lines.append(f"  - {rel} (required by: {required})")
        if drift.is_rollback_file(name):
            lines.append("      rollback file: undoes another migration; never applied on deploy. "
                         "Do not run it unless you are backing that migration out.")
            continue
        if drift.is_destructive(text):
            lines.append("      DESTRUCTIVE: operator-approved only; no apply command is printed. "
                         "Read the file and decide by hand.")
            continue
        r = by_name.get(name)
        if r is None:
            lines.append("      live state: UNKNOWN (the checker did not load this file)")
            continue
        if live is None:
            lines.append(f"      live state: UNKNOWN ({why_unknown}; no answer is not 'applied')")
            if r.status not in ("SKIPPED",):
                lines.append(f"      apply if not yet applied: {r.apply_command()}")
            continue
        if r.status == "APPLIED":
            lines.append("      live state: APPLIED")
        elif r.broken:
            refused = (" -- the services above stay refused by safe_docker_build.sh until it is"
                       if gated else "")
            lines.append(f"      live state: NOT APPLIED{refused}")
            lines.append(f"      {r.summary()}")
            lines.append(f"      apply: {r.apply_command()}")
        elif r.status == "DATA":
            lines.append("      live state: UNVERIFIABLE (data-only; the schema cannot show whether it ran)")
            lines.append(f"      apply if not yet applied: {r.apply_command()}")
        else:
            lines.append(f"      live state: {r.status}")
        for note in r.info:
            lines.append(f"      note: {note}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only problems and verify-manually files")
    ap.add_argument("--file", help="show a single migration's verdict by filename")
    ap.add_argument("--since-days", type=int, default=drift.DEFAULT_WINDOW_DAYS,
                    help="only alarm on migrations changed in the last N days (0 = all)")
    ap.add_argument("--ref", default="HEAD", help="git ref whose history orders the migrations")
    ap.add_argument("--service", help="deploy gate: refuse unless every migration REQUIRED-BY this "
                                      "service is applied (exit 1 missing, 2 unknown)")
    ap.add_argument("--post-pull-report", nargs="+", metavar="PATH",
                    help="post-merge view of these changed migration paths (repo-relative): "
                         "required-by, live state, apply command. Read-only; always exits 0")
    args = ap.parse_args()
    if args.post_pull_report:
        for line in post_pull_report(args.post_pull_report, args.ref):
            print(line)
        return 0
    if args.service:
        return service_gate(args.service, args.ref)

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
