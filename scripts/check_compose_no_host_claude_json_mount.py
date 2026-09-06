#!/usr/bin/env python3
"""Gate: no service compose file may bind-mount the operator's ~/.claude.json.

Why this exists (real incident, 2026-09-06):

orion-harness-governor and orion-hub both mounted
`${HOME}/.claude.json:/root/.claude.json:ro` so Orion's `claude -p`
subprocess would inherit the host's Claude Code config. Two things go wrong
with that single-file bind mount:

1. Claude Code rewrites ~/.claude.json by atomic replace (write temp, rename),
   so the mount pins the inode from container start and silently goes stale.
   Confirmed live: the container's copy was dated 09-03 while the host file
   was newer, so host-side config edits never reached Orion until a restart.

2. The file carries the host-global MCP server list. `caveman` was installed
   on the host on 09-01; its binary does not exist in the container, so every
   Orion session from 09-02 opened with "configured MCP servers failed to
   connect: caveman (ENOENT)" -- and the chat model misread that as a GitHub
   outage and stopped using the GitHub tools it actually had.

The subprocess needs nothing from the host file: spawned with a fresh
CLAUDE_CONFIG_DIR and no host file at all, it answered normally with github,
firecrawl and gitnexus all `connected` (same date). Per-container Claude
state belongs in that container's own config dir / volume.

This is the deterministic gate for that rule, per CLAUDE.md section 4.
"""
from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

# A volume entry whose SOURCE side is a path ending in `.claude.json`,
# whatever prefix it carries (${HOME}, an absolute path, a ${VAR:-default}).
_CLAUDE_JSON_SOURCE = re.compile(r"^\s*-\s*([^:#]*?/\.claude\.json)\s*:")


def find_offenders(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        m = _CLAUDE_JSON_SOURCE.match(line)
        if m:
            out.append((lineno, m.group(1)))
    return out


def self_test() -> int:
    """The gate must bite on the exact line this rule was written against."""
    bad = (
        "services:\n  x:\n    volumes:\n"
        "      - ${HOME}/.fcc:/root/.fcc\n"
        "      - ${HOME}/.claude.json:/root/.claude.json:ro\n"
        "      # - ${HOME}/.claude.json:/root/.claude.json:ro  (commented out is fine)\n"
        "      - harness-claude-config:/root/.claude\n"
    )
    got = find_offenders(bad)
    if got != [(5, "${HOME}/.claude.json")]:
        print(f"compose claude.json-mount gate: SELF-TEST FAIL, got {got!r}")
        return 1
    print("compose claude.json-mount gate: self-test PASS")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()

    offenders: list[tuple[str, int, str]] = []
    composes = sorted(REPO.glob("services/*/docker-compose.yml"))
    for path in composes:
        for lineno, src in find_offenders(path.read_text(encoding="utf-8")):
            offenders.append((str(path.relative_to(REPO)), lineno, src))

    if offenders:
        print("compose claude.json-mount gate: FAIL")
        print("")
        print("  The operator's ~/.claude.json must not be bind-mounted into a container:")
        print("  Claude Code atomically replaces it (the mount goes stale) and it leaks")
        print("  host-global MCP servers into Orion's subprocess (see this script's docstring).")
        print("")
        for rel, lineno, src in offenders:
            print(f"    {rel}:{lineno}  {src}")
        print("")
        print("  Fix: drop the mount. The claude subprocess runs fine from its own config dir.")
        return 1

    print(f"compose claude.json-mount gate: PASS ({len(composes)} compose files, 0 ~/.claude.json mounts)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
