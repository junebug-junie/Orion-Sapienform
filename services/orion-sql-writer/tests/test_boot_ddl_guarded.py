"""Every boot-time schema statement in app/main.py goes through GuardedDDL (2026-10-31).

An ALTER/CREATE left on conn.exec_driver_sql would lock its table on every boot again; a block that
lost its ``ddl = GuardedDDL(...)`` line would raise NameError, which the boot handler swallows while
the whole block rolls back. Both are invisible until a backup overlaps a deploy, so pin them here.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from orion.db.guarded_ddl import COLUMNS_SQL, INDEXES_SQL, TABLES_SQL, GuardedDDL

MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
DDL_START = re.compile(r"^\s*(ALTER|CREATE)\b", re.I)


def _sql(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "x" for v in node.values)
    return None


def _calls(tree, name_check):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and name_check(node.func) and node.args:
            yield node, _sql(node.args[0])


TREE = ast.parse(MAIN.read_text())


def test_no_boot_ddl_bypasses_the_guard():
    bypass = [n.lineno for n, sql in _calls(TREE, lambda f: isinstance(f, ast.Attribute) and f.attr == "exec_driver_sql")
              if sql and DDL_START.match(sql)]
    assert bypass == [], f"ALTER/CREATE on conn.exec_driver_sql at main.py lines {bypass}: use ddl(...)"


def test_every_ddl_call_sits_in_a_block_that_defines_ddl_first():
    for node in ast.walk(TREE):
        if not (isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "attr", "") == "begin"
                for i in node.items)):
            continue
        uses = [c for c in ast.walk(node) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "ddl"]
        if not uses:
            continue
        first = node.body[0]
        assert isinstance(first, ast.Assign) and first.targets[0].id == "ddl", f"block at line {node.lineno}"
        assert getattr(first.value.func, "id", "") == "GuardedDDL", f"block at line {node.lineno}"


def test_every_guarded_statement_is_a_shape_the_guard_can_skip():
    """A new shape would quietly run (and lock) every boot: notice it here instead."""
    calls = list(_calls(TREE, lambda f: isinstance(f, ast.Name) and f.id == "ddl"))
    assert len(calls) >= 100
    unjudged = []
    for node, sql in calls:
        assert sql is not None, f"non-literal SQL at line {node.lineno}"
        tables = {"t"}
        m = re.match(r"\s*ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\S+)", sql, re.I)
        if m:
            tables = {m.group(1).lower()}
        cols = {(t, c) for t in tables for c in re.findall(r"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+(\S+)", sql, re.I)}
        idx = set(re.findall(r"INDEX\s+IF\s+NOT\s+EXISTS\s+(\S+)", sql, re.I))
        tab = set(re.findall(r"TABLE\s+IF\s+NOT\s+EXISTS\s+(\S+)", sql, re.I))
        everything = {TABLES_SQL: [(t,) for t in tables | {x.lower() for x in tab}],
                      COLUMNS_SQL: [(t, c.lower()) for t, c in cols], INDEXES_SQL: [(i.lower(),) for i in idx]}
        if not GuardedDDL(lambda *a: None, lambda q: everything[q]).satisfied(sql):
            unjudged.append((node.lineno, sql.strip()[:80]))
    assert unjudged == []
