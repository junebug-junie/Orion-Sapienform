#!/usr/bin/env python3
"""Deterministic gate: fails if anything directly under scripts/ collides
with a Python stdlib module name.

Why this exists: scripts/platform/ (renamed to scripts/platform_audits/,
2026-08-12 -- see that package's README for the full incident writeup)
silently shadowed stdlib `platform` whenever scripts/ landed on sys.path,
which Python does automatically for any `python3 scripts/<name>.py`
invocation -- the dominant way scripts in this repo actually get invoked
(git hooks, cron jobs, manual runs). The failure mode was a confusing
`AttributeError` deep inside an unrelated import chain (redis's asyncio
submodule -> stdlib uuid.py -> `platform.system()`), not a clean
`ImportError`, and it took an extended live investigation to trace back to
this directory. By the time it was found, ~24 OTHER files across the repo
had already independently discovered and locally worked around this exact
collision (grep `shadows stdlib` or `sys.path.pop(0)` under scripts/ and
orion/) without any of them fixing the actual cause. This gate exists so
the NEXT collision (a future scripts/ addition happening to be named
`json`, `re`, `types`, `enum`, or any other of the ~300 stdlib module
names) gets caught deterministically at review/CI time instead of
rediscovered the same expensive way, one broken script at a time.

Scope: only checks scripts/'s own top-level entries (files and
directories), since scripts/ itself is the directory Python actually
inserts at sys.path[0] for the dominant real invocation shape (`python3
scripts/<name>.py`) -- that's the exact collision this gate protects
against. A same-named collision one level deeper (e.g.
scripts/git_hooks/re.py) would only matter if something is ever invoked
directly as `python3 scripts/git_hooks/<name>.py`; not checked here to
keep this gate thin. If that invocation shape ever becomes real for a
subdirectory, extend find_stdlib_shadows() to recurse into it specifically
-- don't blanket-recurse the whole tree, since most subdirectories (e.g.
scripts/backup/, scripts/tmp/) are never used as sys.path[0] for anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def find_stdlib_shadows(scripts_dir: Path = SCRIPTS_DIR) -> list[str]:
    """Returns the sorted, deduplicated list of top-level entry names under
    scripts_dir that collide with a stdlib module name. Only the two
    shapes Python's import system actually resolves count: a directory
    (package) or a .py file (module) -- both shadow `import <name>` the
    same way scripts/platform/ did. A same-named directory entry that
    ISN'T a real directory (unlikely, but e.g. a broken symlink) or a
    non-.py file (e.g. a stray `platform.sh`) is correctly NOT flagged --
    review finding, 2026-08-13: an earlier version used `entry.name`
    unconditionally for any non-.py entry, which would have flagged a
    hypothetical extensionless file too, even though such a file isn't
    actually importable and therefore doesn't really shadow anything.
    Empty list means clean."""
    stdlib_names = set(sys.stdlib_module_names)
    collisions: set[str] = set()
    for entry in scripts_dir.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            name = entry.name
        elif entry.suffix == ".py":
            name = entry.stem
        else:
            continue
        if name in stdlib_names:
            collisions.add(name)
    return sorted(collisions)


def main() -> int:
    # Pass SCRIPTS_DIR explicitly rather than relying on
    # find_stdlib_shadows()'s default parameter value -- a default is bound
    # once at function-definition time, so a caller (e.g. a test)
    # monkeypatching the module-level SCRIPTS_DIR wouldn't affect an
    # already-bound default. Referencing the global by name here looks it
    # up dynamically at call time instead.
    collisions = find_stdlib_shadows(SCRIPTS_DIR)
    if not collisions:
        print(f"[check_scripts_dir_no_stdlib_shadow] clean -- no stdlib-name collisions under {SCRIPTS_DIR}")
        return 0

    print(f"[check_scripts_dir_no_stdlib_shadow] FOUND {len(collisions)} stdlib-name collision(s) under {SCRIPTS_DIR}:")
    for name in collisions:
        print(f"  - {name}")
    print(
        "\nAny of these will shadow the real stdlib module for ANY script "
        "invoked as `python3 scripts/<name>.py` (Python auto-inserts "
        "scripts/ at sys.path[0] for that invocation shape). See "
        "scripts/platform_audits/README.md for the incident this gate "
        "exists to prevent recurring. Rename the colliding entry."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
