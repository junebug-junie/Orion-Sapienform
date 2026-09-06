#!/usr/bin/env python3
"""Deterministic gate: graphify's scanner must still see this repo's code.

Root cause of the 2026-07-14 "incremental graph refresh silently shrinks the
graph ~95%" incident (root-caused 2026-09-06): graphify 0.9.15 applies a
nested `.gitignore`'s UNANCHORED patterns to the entire remaining directory
walk, not just that file's subtree. One file --
`services/orion-hub/tests/e2e/artifacts/.gitignore`, containing a bare `*` --
therefore ignored every directory os.walk visited after it, the scanner
reported 0 code files at repo root, and the refresh's AST-ownership rule
evicted every code node the scan no longer "owned". Its own log said "kept
26,564 node(s)" while writing 2,485.

Two layers guard it now:

1. `tests/test_gitignore_no_bare_catchall.py` refuses a tracked `.gitignore`
   with a bare catch-all line (`*`, `**`). The anchored form (`/*`) means the
   same thing to git and is confined to its directory by graphify.
2. This script runs graphify's own scanner at repo root and fails when the
   code-file count it returns is implausibly low versus what git tracks. Run
   by `scripts/safe_graphify_update.sh` before every refresh, so a hollow
   scan refuses to run instead of producing a hollow graph.

Exit 0 = scan looks sane. Exit 1 = hollow scan, do not refresh. Exit 0 with a
notice when graphify's python is not installed (nothing to gate).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Extensions graphify has AST extractors for and this repo has in volume.
_CODE_SUFFIXES = (".py", ".js", ".ts", ".sh")
# A real scan sees the large majority of tracked code; 0.5 leaves room for
# graphify's own noise/vendor exclusions without ever accepting "nearly none".
MIN_FRACTION = float(os.getenv("GRAPHIFY_SCAN_MIN_FRACTION", "0.5"))


def _graphify_python() -> str | None:
    exe = Path.home() / ".local/share/uv/tools/graphifyy/bin/python"
    if exe.exists():
        return str(exe)
    found = subprocess.run(["which", "graphify"], capture_output=True, text=True).stdout.strip()
    if found:
        try:
            first = Path(found).read_text(errors="ignore").splitlines()[0]
        except (OSError, IndexError):
            first = ""
        if first.startswith("#!") and Path(first[2:].strip()).exists():
            return first[2:].strip()
    return None


def tracked_code_count(root: Path) -> int:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return sum(1 for line in out.splitlines() if line.endswith(_CODE_SUFFIXES))


def graphify_code_count(py: str, root: Path) -> int:
    code = (
        "from pathlib import Path; from graphify.detect import detect; "
        f"r = detect(Path({str(root)!r})); f = r.get('files', r); print(len(f.get('code', [])))"
    )
    out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-500:])
    return int(out.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(REPO_ROOT), help="checkout to scan (default: this script's repo)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    py = _graphify_python()
    if py is None:
        print("check_graphify_scan_scope: graphify not installed here; nothing to gate.")
        return 0
    tracked = tracked_code_count(root)
    seen = graphify_code_count(py, root)
    floor = int(tracked * MIN_FRACTION)
    print(
        f"check_graphify_scan_scope: graphify sees {seen} code files; "
        f"git tracks {tracked}; floor {floor}"
    )
    if seen < floor:
        print(
            "check_graphify_scan_scope: HOLLOW SCAN -- graphify's scanner is excluding most of "
            "the repo. A refresh now would evict every code node it no longer 'owns'. Look for "
            "a nested .gitignore with a bare catch-all (`*`) -- graphify leaks those across the "
            "whole walk -- and rewrite it anchored (`/*`).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
