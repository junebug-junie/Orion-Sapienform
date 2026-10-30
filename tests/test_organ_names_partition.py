"""Orion's self ledgers stay out of recall until battle-tested (design PR #2888, decision 7): the
organ names (rung 3) and the capacity claims and capacity sentences (rung 4, design
``docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md``, section 6).

Runs in the repo-wide static gates, not the body-atlas workflow: a new reader added anywhere else
in the repo must fail CI even when no body-atlas file changed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
ALLOWED = ("orion/body_atlas/", "orion/dig/", "tests/test_organ_names_partition.py")


def _readers(*terms: str) -> list[str]:
    args = [x for t in terms for x in ("-e", t)]
    proc = subprocess.run(["git", "-C", str(_REPO), "grep", "-l", "-F", *args, "--", "*.py"],
                          capture_output=True, text=True)
    assert proc.returncode in (0, 1), proc.stderr  # 1 = no match; anything else is a broken check
    return proc.stdout.split()


def test_only_the_atlas_and_dig_read_the_organ_names_ledger():
    hits = _readers("organ_names", "body_atlas.names", "body_atlas import names", "names_path")
    assert "orion/body_atlas/names.py" in hits or "orion/dig/__main__.py" in hits  # the grep really ran
    assert [h for h in hits if not h.startswith(ALLOWED)] == []


def test_only_the_atlas_and_dig_read_the_capacity_ledgers():
    hits = _readers("capacity_claims", "capacities.jsonl", "body_atlas.claims", "body_atlas import claims",
                    "import capacities, claims", "claims_path", "capacities_path")
    assert "orion/body_atlas/claims.py" in hits and "orion/dig/__main__.py" in hits  # the grep really ran
    assert [h for h in hits if not h.startswith(ALLOWED)] == []


LEDGER_MODULES = ("names", "claims")  # orion.body_atlas.<module> reads a self ledger


def _ledger_imports(tree) -> list[str]:
    import ast

    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "orion.body_atlas":
            out += [a.name for a in node.names if a.name in LEDGER_MODULES]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out += [node.module for m in LEDGER_MODULES if node.module == f"orion.body_atlas.{m}"]
        elif isinstance(node, ast.Import):
            out += [a.name for a in node.names if a.name in {f"orion.body_atlas.{m}" for m in LEDGER_MODULES}]
    return out


def test_no_module_outside_the_atlas_and_dig_imports_a_ledger_module_in_any_form():
    """The string grep misses `import names, claims` lists and parenthesised imports; the AST does not."""
    import ast

    files = subprocess.run(["git", "-C", str(_REPO), "ls-files", "*.py"], capture_output=True, text=True,
                           check=True).stdout.split()
    assert len(files) > 100  # the listing really ran
    bad = []
    for rel in files:
        if rel.startswith(ALLOWED):
            continue
        try:
            text = (_REPO / rel).read_text(encoding="utf-8")
            if "body_atlas" not in text:  # every import form names the package
                continue
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
            continue
        if _ledger_imports(tree):
            bad.append(rel)
    assert bad == []


def test_the_ast_check_sees_every_import_form():
    import ast

    for src in ("from orion.body_atlas import capacities, claims", "from orion.body_atlas import (\n    names,\n)",
                "import orion.body_atlas.claims", "from orion.body_atlas.names import load"):
        assert _ledger_imports(ast.parse(src)), src
    assert not _ledger_imports(ast.parse("from orion.body_atlas import capacities"))
