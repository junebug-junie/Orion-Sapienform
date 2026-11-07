"""Orion's self ledgers stay out of recall until battle-tested (design PR #2888, decision 7): the
organ names (rung 3), the capacity claims and capacity sentences (rung 4, design
``docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md``, section 6), and the PR
field notes (``docs/superpowers/specs/2026-11-04-pr-field-notes-design.md``, "Storage and partition"),
and the portrait lines (rung 5, ``docs/superpowers/specs/2026-11-06-self-portrait-design.md``).

Runs in the repo-wide static gates, not the body-atlas workflow: a new reader added anywhere else
in the repo must fail CI even when no body-atlas file changed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
ALLOWED = ("orion/body_atlas/", "orion/dig/", "tests/test_organ_names_partition.py")
# The one named exception, for the field-notes ledger ONLY (spec 2026-11-04-pr-field-notes-design.md,
# Patch 2): the automatic half writes code records and checks a note turn's save receipt. It hands back
# ids and booleans, never a note's text (test_the_field_note_writer_never_hands_back_note_text below).
FIELD_NOTE_WRITER = "services/orion-durable-runs/app/pr_field_note_ledger.py"


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


def test_only_the_atlas_and_dig_read_the_pr_field_notes_ledger():
    hits = _readers("pr_field_notes", "body_atlas.field_notes", "body_atlas import field_notes", "field_notes_path")
    assert "orion/body_atlas/field_notes.py" in hits and "orion/dig/__main__.py" in hits  # the grep really ran
    assert FIELD_NOTE_WRITER in hits  # the exception is real, so this check really covers it
    assert [h for h in hits if not h.startswith(ALLOWED) and h != FIELD_NOTE_WRITER] == []


def test_only_the_atlas_and_dig_read_the_portrait_ledger():
    hits = _readers("portrait.jsonl", "body_atlas.portrait", "body_atlas import portrait", "portrait_path")
    assert "orion/dig/__main__.py" in hits  # the grep really ran
    assert [h for h in hits if not h.startswith(ALLOWED)] == []


LEDGER_MODULES = ("names", "claims", "field_notes", "portrait")  # orion.body_atlas.<module> reads a self ledger


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
        found = _ledger_imports(tree)
        if rel == FIELD_NOTE_WRITER:  # may import field_notes, and nothing else of the self ledgers
            found = [m for m in found if m not in ("field_notes", "orion.body_atlas.field_notes")]
        if found:
            bad.append(rel)
    assert bad == []


def test_the_field_note_writer_never_hands_back_note_text():
    """The exception reads records only to answer "is there one" / "is this id a note": every use of a
    record's ``.note`` is an ``is not None`` test, so no note text can leave that module."""
    import ast

    tree = ast.parse((_REPO / FIELD_NOTE_WRITER).read_text(encoding="utf-8"))
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "note"]
    assert uses  # the check really looked at something
    ok = {id(c.left) for c in ast.walk(tree) if isinstance(c, ast.Compare) and len(c.ops) == 1
          and isinstance(c.ops[0], ast.IsNot) and isinstance(c.comparators[0], ast.Constant)
          and c.comparators[0].value is None}
    assert [ast.unparse(u) for u in uses if id(u) not in ok] == []


def test_the_field_note_writer_returns_only_ids_and_booleans():
    """Every public function of the exception declares an id/boolean-shaped return, and no ``return``
    hands back a loaded record list, a live map or a bare variable (review 2026-11-04: the ``.note``
    check alone would miss ``return records``)."""
    import ast

    allowed = {"bool", "str", "str | None", "None", "RecordRef", "dict[int, RecordRef]"}
    tree = ast.parse((_REPO / FIELD_NOTE_WRITER).read_text(encoding="utf-8"))
    public = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and not f.name.startswith("_")]
    assert len(public) >= 5  # the check really looked at the module's API
    assert {f.name: ast.unparse(f.returns) if f.returns else None for f in public
            if (ast.unparse(f.returns) if f.returns else None) not in allowed} == {}
    bad = []
    for ret in (n for n in ast.walk(tree) if isinstance(n, ast.Return) and n.value is not None):
        v = ret.value
        if isinstance(v, ast.Name) or (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                                       and v.func.attr in ("load", "live", "model_dump", "model_dump_json")):
            bad.append(ast.unparse(ret))
    assert bad == []


def test_the_ast_check_sees_every_import_form():
    import ast

    for src in ("from orion.body_atlas import capacities, claims", "from orion.body_atlas import (\n    names,\n)",
                "import orion.body_atlas.claims", "from orion.body_atlas.names import load"):
        assert _ledger_imports(ast.parse(src)), src
    assert _ledger_imports(ast.parse("from orion.body_atlas import field_notes as fn"))
    assert _ledger_imports(ast.parse("from orion.body_atlas import portrait"))
    assert not _ledger_imports(ast.parse("from orion.body_atlas import capacities"))
