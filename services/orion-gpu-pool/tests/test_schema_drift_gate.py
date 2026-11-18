"""Drift gate: the pool's boot self-heal list, the operator migration files and the columns the
store writes must agree, column for column.

The pool adds missing additive columns itself at boot (app/store.py BOOT_ADDITIVE_COLUMNS) so a pool
deployed before its migration keeps serving instead of crash-looping (three total LLM outages:
2026-09-26 v2, 2026-09-30 v3). That only stays safe if the boot list never drifts from the files an
operator runs, and never grows anything non-additive. No database needed."""
from __future__ import annotations

import re
from pathlib import Path

from app.store import BOOT_ADDITIVE_COLUMNS, CARD_COLUMNS, LEASE_COLUMNS, REQUIRED_COLUMNS

SQL_DB = Path(__file__).resolve().parents[3] / "services/orion-sql-db"
V1 = SQL_DB / "manual_migration_gpu_pool_v1.sql"
ADD_COLUMN = re.compile(r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+(\w+)\s+([^;]+);", re.I)


def _sql(path: Path) -> str:
    return "\n".join(l.split("--", 1)[0] for l in path.read_text().splitlines())


def _norm(ddl: str) -> str:
    return " ".join(ddl.split()).lower()


def _v1_columns() -> set[tuple[str, str]]:
    out = set()
    for table, body in re.findall(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s*\((.*?)\n\);", _sql(V1), re.S | re.I):
        for line in body.splitlines():
            if m := re.match(r"\s*(\w+)\s+\w", line):
                out.add((table, m.group(1)))
    return out


def _migration_columns() -> dict[tuple[str, str], tuple[str, str]]:
    """(table, column) -> (normalized ddl, file) for every ADD COLUMN in any manual_migration_gpu_pool_*.sql."""
    out = {}
    for path in sorted(SQL_DB.glob("manual_migration_gpu_pool_*.sql")):
        sql = _sql(path)
        # an ADD COLUMN in any other spelling would slip past the regex -- and past this gate
        assert len(re.findall(r"\bADD\s+COLUMN\b", sql, re.I)) == len(ADD_COLUMN.findall(sql)), \
            f"{path.name}: write ADD COLUMN as `ALTER TABLE t ADD COLUMN IF NOT EXISTS c <type>;`"
        for table, column, ddl in ADD_COLUMN.findall(sql):
            assert (table, column) not in out, f"{table}.{column} added by two migrations"
            out[(table, column)] = (_norm(ddl), path.name)
    return out


def test_the_parsers_find_what_they_must():
    """A regex that matches nothing would make every other check here pass vacuously."""
    assert ("gpu_pool_leases", "lease_id") in _v1_columns() and ("gpu_pool_cards", "updated_by") in _v1_columns()
    assert len(_migration_columns()) >= 9


def test_every_column_the_store_writes_is_created_by_v1_or_the_boot_list():
    boot = {(c.table, c.column) for c in BOOT_ADDITIVE_COLUMNS}
    v1 = _v1_columns()
    for table, cols in (("gpu_pool_leases", LEASE_COLUMNS), ("gpu_pool_cards", CARD_COLUMNS)):
        for col in cols:
            assert (table, col) in v1 or (table, col) in boot, \
                f"{table}.{col} is written by the store but no migration/boot entry creates it"
    assert set(REQUIRED_COLUMNS) == {"gpu_pool_leases", "gpu_pool_cards"}


def test_boot_list_and_operator_migrations_match_exactly():
    """Both directions: a new ADD COLUMN migration without a boot entry (the pool would refuse to
    boot on it) fails here, and so does a boot entry no operator file carries, or a type drift."""
    files = _migration_columns()
    boot = {(c.table, c.column): (_norm(c.ddl), c.migration) for c in BOOT_ADDITIVE_COLUMNS}
    assert boot == files


def test_boot_ddl_is_additive_only():
    """Nullable, or NOT NULL with a constant default: an instant catalog change with no rewrite.
    The python default must be what Postgres fills in, since a degraded pool reads it for rows."""
    for c in BOOT_ADDITIVE_COLUMNS:
        ddl = _norm(c.ddl)
        assert not re.search(r"\b(drop|rename|using|primary|unique|references|generated)\b", ddl), c
        m = re.search(r"\bdefault\s+(.+)$", ddl)
        if "not null" in ddl:
            assert m, f"{c.table}.{c.column}: NOT NULL without a DEFAULT would fail on a table with rows"
        if m:
            literal = m.group(1).replace("not null", "").strip()
            assert re.fullmatch(r"-?\d+|'[^']*'|true|false", literal), f"{c}: default must be a constant"
            assert str(c.default).lower() == literal.strip("'"), c
        else:
            assert c.default is None, c
