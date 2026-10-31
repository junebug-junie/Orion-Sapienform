"""Boot-time schema statements that skip what the catalog already has.

``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` takes an ACCESS EXCLUSIVE lock even when the column is
already there, and ``CREATE INDEX IF NOT EXISTS`` a SHARE lock before it checks the name. Run on
every start, they queue behind any long reader or writer -- live 2026-10-31 a 27-minute pg_dump held
topic-foundry's startup at its first ALTER for 11+ minutes. ``GuardedDDL`` reads the catalog once and
skips the statements it shows are already satisfied, so a current schema takes no table lock.

Only shapes it can judge exactly are ever skipped: ADD COLUMN IF NOT EXISTS (one or several clauses
and nothing else), CREATE [UNIQUE] INDEX IF NOT EXISTS, CREATE TABLE IF NOT EXISTS. Anything else
runs as before. Names are matched unqualified in ``current_schema()``.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable

_ALTER = re.compile(r"^\s*ALTER\s+TABLE\s+(IF\s+EXISTS\s+)?(?:ONLY\s+)?(\S+)\s+(.*?)\s*;?\s*$", re.I | re.S)
_ADD_COLUMN = re.compile(r"^ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+(\S+)\s", re.I)
_CREATE_INDEX = re.compile(r"^\s*CREATE\s+(?:UNIQUE\s+)?INDEX\s+IF\s+NOT\s+EXISTS\s+(\S+)", re.I)
_CREATE_TABLE = re.compile(r"^\s*CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\S+)", re.I)

TABLES_SQL = "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
COLUMNS_SQL = ("SELECT table_name, column_name FROM information_schema.columns "
               "WHERE table_schema = current_schema()")
INDEXES_SQL = "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"


def _name(raw: str) -> str:
    """Postgres folds unquoted identifiers to lower case and keeps quoted ones as written."""
    last = raw.strip().rstrip(";(").split(".")[-1]
    return last[1:-1] if len(last) > 1 and last.startswith('"') and last.endswith('"') else last.lower()


def _clauses(body: str) -> list[str]:
    """Top-level comma-separated clauses of an ALTER TABLE body (commas inside parentheses ignored)."""
    out, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


class GuardedDDL:
    """``ddl(sql)`` runs ``sql`` through ``execute`` unless the catalog read at construction shows it
    is already satisfied. Returns True when it ran. ``ran`` / ``skipped`` count both for a log line."""

    def __init__(self, execute: Callable[..., Any], query: Callable[[str], Iterable[tuple]]):
        self._execute = execute
        # catalog names as stored (unquoted ones are already lower case there)
        self.tables = {r[0] for r in query(TABLES_SQL)}
        self.columns = {(r[0], r[1]) for r in query(COLUMNS_SQL)}
        self.indexes = {r[0] for r in query(INDEXES_SQL)}
        self._created: set[str] = set()   # tables made in this run: their columns were never read
        self.ran = 0
        self.skipped = 0

    def satisfied(self, sql: str) -> bool:
        if ";" in sql.strip().rstrip(";") or "--" in sql or "/*" in sql:
            return False              # several statements or comments: only the first could be judged
        m = _CREATE_INDEX.match(sql)
        if m:
            return _name(m.group(1)) in self.indexes
        m = _CREATE_TABLE.match(sql)
        if m:
            return _name(m.group(1)) in self.tables
        m = _ALTER.match(sql)
        if not m:
            return False
        table = _name(m.group(2))
        added = [_ADD_COLUMN.match(c + " ") for c in _clauses(m.group(3))]
        if not added or not all(added) or table in self._created or table not in self.tables:
            # A table missing from this snapshot is left to Postgres: ALTER TABLE IF EXISTS on it
            # takes no lock anyway, and the snapshot's schema might not be the one search_path finds.
            return False
        return all((table, _name(a.group(1))) in self.columns for a in added)

    def __call__(self, sql: str, *args: Any) -> bool:
        if self.satisfied(sql):
            self.skipped += 1
            return False
        self._execute(sql, *args)
        self.ran += 1
        m = _CREATE_TABLE.match(sql)
        if m and _name(m.group(1)) not in self.tables:
            self.tables.add(_name(m.group(1)))
            self._created.add(_name(m.group(1)))
        m = _CREATE_INDEX.match(sql)
        if m:
            self.indexes.add(_name(m.group(1)))
        return True
