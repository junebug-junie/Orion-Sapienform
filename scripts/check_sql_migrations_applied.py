#!/usr/bin/env python3
"""Report which hand-applied SQL migrations actually reached the live database.

WHY THIS EXISTS
`services/orion-sql-db/*.sql` is 78 files that are applied BY HAND, by whoever wrote them,
with a `docker exec -i ... psql < file` in a commit message. Nothing tracks which ones ran.
There is no migration table, no version stamp, no ordering guarantee, and no way to answer
"is this repo's declared schema the schema we are actually running" other than opening each
file and checking by hand.

That is the exact shape of problem CLAUDE.md says to turn into a script rather than a
reminder. A migration that was written, reviewed, merged, and never applied looks identical
in git to one that is live.

WHAT IT CHECKS, AND WHAT IT DELIBERATELY DOES NOT
Three statement forms are declarative -- the object either exists in the live database or it
does not -- and they cover the overwhelming majority of what these files do:

    CREATE [UNIQUE] INDEX [CONCURRENTLY] [IF NOT EXISTS] <name> ON <table> ...
    CREATE TABLE [IF NOT EXISTS] <name> ...
    ALTER TABLE <table> ADD COLUMN [IF NOT EXISTS] <column> ...

Everything else -- INSERT, UPDATE, DELETE, DROP, backfills, `ALTER TABLE ... SET (...)`,
functions, views -- is NOT checked, because "did this backfill run" is not answerable from
the presence of a schema object. Those are counted and reported as UNCHECKED per file, never
silently skipped. A file whose statements are all unchecked reports as UNKNOWN, not as
applied. Reporting a false clean bill of health would be worse than having no gate at all.

INVALID INDEXES ARE A FAILURE, NOT A PASS. An interrupted `CREATE INDEX CONCURRENTLY` leaves
an index that exists in pg_class but has `indisvalid = false`. It is invisible to
`IF NOT EXISTS` (so re-running the migration is a silent no-op) and unusable by the planner
(so queries silently fall back to seq scans). CLAUDE.md's own migration instructions warn
about this. Present-but-invalid is reported as INVALID and exits non-zero.

USAGE
    python3 scripts/check_sql_migrations_applied.py                 # human summary
    python3 scripts/check_sql_migrations_applied.py --json          # machine readable
    python3 scripts/check_sql_migrations_applied.py --file X.sql    # one migration
    python3 scripts/check_sql_migrations_applied.py --quiet         # only problems

Exit codes: 0 = everything checkable is applied; 1 = something is missing or invalid;
2 = could not connect to the database (NOT the same as "drift found", and deliberately a
different code so CI cannot mistake an infra failure for a pass).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DIR = REPO_ROOT / "services" / "orion-sql-db"

# Statement splitting is deliberately crude but comment-aware: these files carry very long
# `--` headers that contain the words CREATE INDEX and ALTER TABLE in prose. Stripping line
# comments first is what stops the checker from inventing objects out of documentation.
_LINE_COMMENT = re.compile(r"--[^\n]*")

# Two escape hatches, both of which must be written INTO the migration file. That is the
# point: a permanently-red gate is a gate people learn to ignore, but an exception that lives
# in a commit message or someone's head is not an exception, it is drift with an alibi.
#
# SUPERSEDED-BY names a later migration that intentionally removed what this one created --
# e.g. manual_migration_proposal_frame_v1.sql creates
# idx_substrate_proposal_frames_source_self_state, and *_v2_drop_self_state.sql drops it. The
# v1 file is correct history; its objects are correctly absent. The named file must exist, or
# the marker itself is reported as drift.
#
# NOT-A-MIGRATION is for files that live in this directory but were never meant to be applied
# -- pg_dump output, scratch schema captures. They are skipped entirely.
_SUPERSEDED = re.compile(r"--\s*ORION-MIGRATION-SUPERSEDED-BY:\s*(?P<file>\S+)", re.I)
_NOT_A_MIGRATION = re.compile(r"--\s*ORION-MIGRATION-NOT-A-MIGRATION:\s*(?P<why>.+)", re.I)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)

_INDEX = re.compile(
    r"\bcreate\s+(?:unique\s+)?index\s+(?:concurrently\s+)?(?:if\s+not\s+exists\s+)?"
    r"(?P<name>[a-z_][a-z0-9_$]*)\s+on\s+(?:only\s+)?(?P<table>[a-z_][a-z0-9_$.]*)",
    re.I,
)
_TABLE = re.compile(
    r"\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?(?P<name>[a-z_][a-z0-9_$.]*)", re.I
)
_COLUMN = re.compile(
    r"\balter\s+table\s+(?:if\s+exists\s+)?(?P<table>[a-z_][a-z0-9_$.]*)\s+"
    r"add\s+column\s+(?:if\s+not\s+exists\s+)?(?P<name>[a-z_][a-z0-9_$]*)",
    re.I,
)
# Anything matching one of these is a real statement we are choosing not to verify.
_UNCHECKABLE = re.compile(
    r"\b(insert\s+into|update\s+\w|delete\s+from|drop\s+|alter\s+table\s+\S+\s+"
    r"(?:set|drop|alter|rename)|create\s+or\s+replace|create\s+extension|create\s+type|"
    r"create\s+materialized\s+view|create\s+view|create\s+function|analyze|vacuum|reindex)\b",
    re.I,
)


@dataclass
class Finding:
    kind: str          # index | table | column
    name: str
    table: str | None
    status: str        # applied | missing | invalid
    detail: str = ""


@dataclass
class FileReport:
    path: str
    findings: list[Finding] = field(default_factory=list)
    unchecked: int = 0
    superseded_by: str | None = None
    not_a_migration: str | None = None
    marker_error: str | None = None

    @property
    def status(self) -> str:
        if self.marker_error:
            return "MISSING"  # a marker pointing at nothing is drift in its own right
        if self.not_a_migration:
            return "SKIPPED"
        if any(f.status == "invalid" for f in self.findings):
            return "INVALID"
        if any(f.status == "missing" for f in self.findings):
            # Superseded files are EXPECTED to have missing objects. An applied one is the
            # surprise -- it means the later migration did not actually run.
            return "SUPERSEDED" if self.superseded_by else "MISSING"
        if self.findings:
            return "APPLIED"
        return "UNKNOWN"  # nothing declarative to check -- NOT the same as applied


def strip_comments(sql: str) -> str:
    return _LINE_COMMENT.sub(" ", _BLOCK_COMMENT.sub(" ", sql))


def parse_migration(sql: str) -> tuple[list[tuple[str, str, str | None]], int]:
    """Return ([(kind, name, table)], unchecked_statement_count)."""
    body = strip_comments(sql)
    wanted: list[tuple[str, str, str | None]] = []
    seen: set[tuple[str, str]] = set()

    for m in _INDEX.finditer(body):
        key = ("index", m.group("name").lower())
        if key not in seen:
            seen.add(key)
            wanted.append(("index", key[1], m.group("table").split(".")[-1].lower()))
    for m in _TABLE.finditer(body):
        key = ("table", m.group("name").split(".")[-1].lower())
        if key not in seen:
            seen.add(key)
            wanted.append(("table", key[1], None))
    for m in _COLUMN.finditer(body):
        table = m.group("table").split(".")[-1].lower()
        key = ("column", f"{table}.{m.group('name').lower()}")
        if key not in seen:
            seen.add(key)
            wanted.append(("column", m.group("name").lower(), table))

    unchecked = len(_UNCHECKABLE.findall(body))
    return wanted, unchecked


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


def load_live_state(conn) -> tuple[dict[str, bool], set[str], set[str]]:
    """(index name -> indisvalid, table names, "table.column" names)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.relname, i.indisvalid FROM pg_class c "
            "JOIN pg_index i ON i.indexrelid = c.oid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname NOT IN ('pg_catalog','information_schema')"
        )
        indexes = {r[0].lower(): bool(r[1]) for r in cur.fetchall()}
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog','information_schema')"
        )
        tables = {r[0].lower() for r in cur.fetchall()}
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema NOT IN ('pg_catalog','information_schema')"
        )
        columns = {f"{r[0].lower()}.{r[1].lower()}" for r in cur.fetchall()}
    return indexes, tables, columns


def check(paths: list[Path], indexes, tables, columns) -> list[FileReport]:
    reports = []
    for path in paths:
        raw = path.read_text(errors="replace")
        wanted, unchecked = parse_migration(raw)
        report = FileReport(path=str(path.relative_to(REPO_ROOT)), unchecked=unchecked)

        nam = _NOT_A_MIGRATION.search(raw)
        if nam:
            report.not_a_migration = nam.group("why").strip()
            reports.append(report)
            continue

        sup = _SUPERSEDED.search(raw)
        if sup:
            named = sup.group("file").strip()
            if not (MIGRATION_DIR / Path(named).name).exists():
                report.marker_error = (
                    f"SUPERSEDED-BY names {named!r}, which does not exist in "
                    f"{MIGRATION_DIR.name}/"
                )
                reports.append(report)
                continue
            report.superseded_by = named
        for kind, name, table in wanted:
            if kind == "index":
                if name not in indexes:
                    report.findings.append(Finding(kind, name, table, "missing"))
                elif not indexes[name]:
                    report.findings.append(Finding(
                        kind, name, table, "invalid",
                        "exists but indisvalid=false -- an interrupted CREATE INDEX "
                        "CONCURRENTLY. IF NOT EXISTS will not rebuild it and the planner "
                        "will not use it. DROP INDEX and re-run.",
                    ))
                else:
                    report.findings.append(Finding(kind, name, table, "applied"))
            elif kind == "table":
                report.findings.append(Finding(
                    kind, name, None, "applied" if name in tables else "missing"))
            else:
                key = f"{table}.{name}"
                report.findings.append(Finding(
                    kind, name, table, "applied" if key in columns else "missing"))
        reports.append(report)
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only files with problems")
    ap.add_argument("--file", help="check a single migration by filename")
    args = ap.parse_args()

    paths = sorted(MIGRATION_DIR.glob("*.sql"))
    if args.file:
        paths = [p for p in paths if p.name == args.file or str(p).endswith(args.file)]
        if not paths:
            print(f"no migration matching {args.file!r}", file=sys.stderr)
            return 2
    if not paths:
        print(f"no .sql files under {MIGRATION_DIR}", file=sys.stderr)
        return 2

    conn = connect()
    try:
        indexes, tables, columns = load_live_state(conn)
    finally:
        conn.close()

    reports = check(paths, indexes, tables, columns)
    bad = [r for r in reports if r.status in ("MISSING", "INVALID")]

    if args.json:
        print(json.dumps([asdict(r) | {"status": r.status} for r in reports], indent=2))
        return 1 if bad else 0

    by_status: dict[str, int] = {}
    for r in reports:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        if args.quiet and r.status not in ("MISSING", "INVALID"):
            continue
        marks = {"APPLIED": "ok", "MISSING": "MISSING", "INVALID": "INVALID",
                 "UNKNOWN": "unknown", "SUPERSEDED": "superseded", "SKIPPED": "skipped"}
        line = f"[{marks[r.status]:>10}] {r.path}"
        if r.unchecked:
            line += f"  (+{r.unchecked} unchecked statement(s))"
        if r.superseded_by:
            line += f"  (superseded by {Path(r.superseded_by).name})"
        if r.not_a_migration:
            line += f"  ({r.not_a_migration})"
        print(line)
        if r.marker_error:
            print(f"            {r.marker_error}")
        if r.status == "SUPERSEDED":
            continue
        for f in r.findings:
            if f.status == "applied":
                continue
            where = f" on {f.table}" if f.table else ""
            print(f"            {f.status.upper()} {f.kind} {f.name}{where}")
            if f.detail:
                print(f"              {f.detail}")

    print()
    print(f"{len(reports)} migration file(s): " + ", ".join(
        f"{v} {k.lower()}" for k, v in sorted(by_status.items())))
    unknown = [r for r in reports if r.status == "UNKNOWN"]
    if unknown:
        print(f"{len(unknown)} file(s) contain nothing declaratively checkable "
              f"(backfills, DROPs, storage params). UNKNOWN is not 'applied'.")
    if bad:
        print(f"\n{len(bad)} migration file(s) are NOT fully applied to the live database.")
        return 1
    print("Every declaratively checkable object in every migration is present and valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
