#!/usr/bin/env python3
"""Fail when a tuned env key's value is copied somewhere that has drifted.

Some env keys are load-bearing numbers that get restated in several places --
the service's `.env_example`, its compose default, a `Field(...)` default, and
prose in a comment or README explaining a budget derived from them. Every
restatement is a copy, and copies drift.

Measured, 2026-08-26: `HARNESS_FCC_TIMEOUT_SEC` was raised to 1600 in the live
`.env` while `.env_example`, the compose default, the governor's own
`settings.py` default, two Hub comments, `orion/llm/routes.py` and the
llm-gateway README all still said 900. Nothing failed, because nothing was
checking. That number is now stamped into the FCC sandbox at spawn time
(`orion/harness/fcc_motor.py:_build_subprocess_env`) so a turn can read its own
deadline -- which turns a stale copy from something operators misread into
something Orion is actively told.

The rule is NOT "state it once". Code and compose defaults have to exist. The
rule is ONE OWNER: every literal for the key, anywhere, must equal the value in
the owner file. The gate never hardcodes the number itself, so retuning the key
in its owner is a one-line change that keeps this passing.

Usage:
    python scripts/check_env_key_single_source.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent

# key -> the file whose value is authoritative.
OWNERS: dict[str, str] = {
    "HARNESS_FCC_TIMEOUT_SEC": "services/orion-harness-governor/.env_example",
}

# Line-level opt-out for text that quotes a value deliberately.
SAMPLE_MARKER = "env-key-single-source: sample"

SCANNED_SUFFIXES = {".py", ".md", ".yml", ".yaml"}
SCANNED_NAMES = {".env_example"}

# Historical records, deliberately frozen at what was true when written. A PR
# report or a design spec that silently rewrote itself when a config changed
# would be worse than a stale one.
EXCLUDED_PREFIXES = (
    "docs/superpowers/pr-reports/",
    "docs/superpowers/specs/",
    ".git/",
    "node_modules/",
    "graphify-out/",
    ".worktrees/",
    ".claude/worktrees/",
)


def _iter_files() -> Iterator[Path]:
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        if path.suffix in SCANNED_SUFFIXES or path.name in SCANNED_NAMES:
            yield path


def _same_number(a: str, b: str) -> bool:
    """`900`, `900.0` and `900s` are the same value stated three ways."""
    try:
        return float(a.rstrip("s")) == float(b.rstrip("s"))
    except ValueError:
        return a == b


def _literals(text: str, key: str) -> Iterator[tuple[int, str]]:
    """Every place this text pins the key to a number.

    Deliberately narrow: the number must follow the key directly, through `=`
    or compose's `:-`. Prose like "KEY + OTHER_KEY=300" names the key without
    pinning it and is not a copy of its value.
    """
    patterns = (
        rf"{re.escape(key)}\s*[:]?[-=]\s*([0-9]+(?:\.[0-9]+)?s?)",
        rf"Field\(\s*([0-9]+(?:\.[0-9]+)?)[^)]*alias=[\"']{re.escape(key)}[\"']",
    )
    lines = text.splitlines()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            line = text.count("\n", 0, match.start()) + 1
            # An inline opt-out for text that quotes a value on purpose -- this
            # gate's own fixtures, or a doc showing what drift looks like.
            # Marker rather than a path exclusion so that a *real* stale default
            # sitting in some other test file is still caught.
            if 0 < line <= len(lines) and SAMPLE_MARKER in lines[line - 1]:
                continue
            yield line, match.group(1)


def main() -> int:
    failures: list[str] = []
    for key, owner_rel in OWNERS.items():
        owner = REPO_ROOT / owner_rel
        if not owner.is_file():
            failures.append(f"{key}: owner file {owner_rel} does not exist")
            continue
        owned = list(_literals(owner.read_text(encoding="utf-8"), key))
        if not owned:
            failures.append(f"{key}: owner {owner_rel} states no value for it")
            continue
        expected = owned[0][1]

        for path in _iter_files():
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel == owner_rel:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if key not in text:
                continue
            for line, found in _literals(text, key):
                if not _same_number(found, expected):
                    failures.append(
                        f"{rel}:{line}: {key} stated as {found}, but "
                        f"{owner_rel} says {expected}"
                    )

    if failures:
        print("Env key drift -- one owner per tuned key:\n", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nFix the copy, not the owner, unless you are deliberately "
            "retuning the key.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(OWNERS)} owned env key(s), no drifted copies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
