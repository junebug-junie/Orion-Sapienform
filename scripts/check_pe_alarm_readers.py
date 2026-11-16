#!/usr/bin/env python3
"""Every reader of node prediction_error values knows that some of those nodes are alarms (#3307).

node:substrate.cabinet / .strain / .street are ALARMS, not prediction errors (PE suite rebuild,
"Alarms move out of the PE group"). They still travel on the node's ``prediction_error`` field and in
``substrate_node_prediction_error_history``, because the attention contests and strain news read them
there. So a new reader that ranks, averages or thresholds prediction_error values across nodes, and
forgets ``is_alarm_node``, reads a firing as a surprise size. This gate makes that a CI failure
instead of a review hope.

A Python file "touches" prediction_error when it has the quoted string ``"prediction_error"`` anywhere
(a dict key, ``.get``, ``getattr``, a constant, a channel name, an SQL JSON path) or ``.prediction_error``
or names the history table. Broad on purpose: the field channel of the same name carries alarms too
(``orion/field/pressure.py`` max-merged them across nodes until #3307), and readers there address it by
name. Each such file must be one of:

- **filters** (automatic): it uses ``is_alarm_node`` or ``ALARM_NODE_IDS`` in CODE (comments and strings
  do not count). This means the author decided how alarms are treated, not a proof the file drops them.
- **registered** in ``config/substrate/prediction_error_readers.yaml`` with a class and a reason:
  ``alarm_aware`` (reads alarms on purpose, as alarms), ``filters_via`` (skips alarms through a helper
  it imports -- name it), ``single_node`` (reads only named non-alarm nodes), ``not_a_reader`` (writes,
  stores, defines, transports or decodes the value; never reads it as a size across nodes).

It also fails on a registry entry whose file is gone or no longer touches prediction_error (a stale
entry hides nothing today but would silently cover a new reader at the same path tomorrow).

    python scripts/check_pe_alarm_readers.py            # repo root; exit 1 on any problem
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path("config/substrate/prediction_error_readers.yaml")
SCAN_ROOTS = ("orion", "services", "scripts")
TOUCH = re.compile(r"substrate_node_prediction_error_history|[\"']prediction_error[\"']|\.prediction_error\b")
FILTERS = re.compile(r"\bis_alarm_node\b|\bALARM_NODE_IDS\b")
CLASSES = ("alarm_aware", "filters_via", "single_node", "not_a_reader")
SKIP_PARTS = {"tests", "test", "evals", "__pycache__", "node_modules", ".venv"}


def _is_test(rel: Path) -> bool:
    return bool(SKIP_PARTS & set(rel.parts)) or rel.name.startswith("test_") or rel.name == "conftest.py"


def _code_only(text: str) -> str:
    """The source with comments and string literals removed (a mention in a docstring is not a filter)."""
    keep = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING) and not getattr(tokenize, "FSTRING_MIDDLE", -1) == tok.type:
                keep.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return ""            # fail closed: an unparseable file never auto-passes on a comment; register it
    return " ".join(keep)


def touching_files(root: Path) -> dict[str, bool]:
    """rel path -> filters automatically, for every non-test .py file that touches prediction_error."""
    out: dict[str, bool] = {}
    for top in SCAN_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(root)
            if _is_test(rel):
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            if TOUCH.search(text):
                out[rel.as_posix()] = bool(FILTERS.search(_code_only(text)))
    return out


def load_registry(root: Path) -> dict:
    doc = yaml.safe_load((root / REGISTRY).read_text(encoding="utf-8")) or {}
    if doc.get("version") != 1:
        raise SystemExit(f"{REGISTRY}: version must be 1")
    return doc


def check(root: Path) -> list[str]:
    doc = load_registry(root)
    problems: list[str] = []
    excluded = []
    for e in doc.get("excluded_paths") or []:
        path = str(e.get("path") or "").strip()
        if not str(e.get("reason") or "").strip():
            problems.append(f"excluded path {path!r} has no reason")
        if not path.endswith("/") or path.rstrip("/") in SCAN_ROOTS or not (root / path).is_dir():
            problems.append(f"excluded path {path!r} must be an existing directory ending in '/', "
                            f"narrower than a whole scan root ({', '.join(SCAN_ROOTS)})")
            continue
        excluded.append(path)
    files = doc.get("files") or {}
    for path, entry in files.items():
        cls = (entry or {}).get("class")
        if cls not in CLASSES:
            problems.append(f"{path}: class {cls!r} is not one of {', '.join(CLASSES)}")
        if not str((entry or {}).get("reason") or "").strip():
            problems.append(f"{path}: no reason")
    touching = touching_files(root)
    for path, auto in touching.items():
        if auto or path in files or any(path.startswith(x) for x in excluded):
            continue
        problems.append(
            f"{path} reads node prediction_error but neither skips alarms (is_alarm_node / ALARM_NODE_IDS) "
            f"nor is registered in {REGISTRY}. Fix: skip alarm nodes where it treats the value as a surprise "
            f"size, or register it with a class ({', '.join(CLASSES)}) and a reason.")
    for path in files:
        if path not in touching:
            why = "no longer exists" if not (root / path).exists() else "no longer touches prediction_error"
            problems.append(f"{path}: registered but {why}; remove the stale entry from {REGISTRY}")
        elif touching[path]:
            problems.append(f"{path}: registered but already filters alarms itself; remove the entry")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    args = ap.parse_args(argv)
    problems = check(args.root)
    if problems:
        print(f"check_pe_alarm_readers: FAIL ({len(problems)} problem(s))")
        for p in problems:
            print(f"  - {p}")
        print("CLAUDE.md 'Deterministic gates': alarms ride the prediction_error field (#3307); every reader "
              "must say how it treats them.")
        return 1
    n = len(touching_files(args.root))
    print(f"check_pe_alarm_readers: OK ({n} file(s) touch prediction_error, all accounted for)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
