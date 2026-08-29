#!/usr/bin/env python3
"""Gate: every SystemHealthV1(...) construction must supply the required fields.

`SystemHealthV1` (orion/schemas/telemetry/system_health.py) requires `boot_id` and
`last_seen_ts`. Every producer in this repo builds it inside a heartbeat loop wrapped
in `try/except Exception: logger.warning(...)`, so omitting a required field does not
crash the service -- it logs once per tick and sleeps. The service stays up, healthy
by every other measure, and publishes no heartbeat at all.

Confirmed live 2026-08-29: `orion-gpu-cluster-power` had been failing every 30s tick
indefinitely and nothing downstream noticed. Auditing the rest found
`orion-bus-tap` and `orion-rag` with the identical defect, and
`services/orion-whisper-tts/app/main.py` carrying a `# FIX: Added boot_id and
last_seen_ts to satisfy SystemHealthV1 schema` comment -- someone hit this before and
fixed exactly one service. That is the signature of a bug class that needs a gate
rather than another fix (CLAUDE.md: "Deterministic gates over repeated yelling").

Deliberately an AST check, not a regex: it parses each call site and inspects the
keyword arguments actually passed, so reformatting, comments and argument order do
not affect it. `**kwargs` splats are reported as unverifiable rather than silently
passed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = "SystemHealthV1"
# Required (no default) on the model. Kept explicit rather than introspected so the
# gate fails loudly if someone relaxes the schema without revisiting this list.
REQUIRED_KWARGS = {"service", "boot_id", "last_seen_ts"}
SEARCH_ROOTS = ("services", "orion")
SKIP_PARTS = {".git", "__pycache__", "node_modules", ".venv", ".worktrees", "tests", "evals"}


def _iter_python_files():
    for root in SEARCH_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if SKIP_PARTS & set(path.parts):
                continue
            yield path


def _check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{path.relative_to(REPO_ROOT)}: could not parse ({exc})"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != TARGET:
            continue

        rel = path.relative_to(REPO_ROOT)
        if any(kw.arg is None for kw in node.keywords):
            problems.append(
                f"{rel}:{node.lineno}: {TARGET}(**kwargs) -- cannot verify required "
                f"fields statically; pass them explicitly"
            )
            continue

        supplied = {kw.arg for kw in node.keywords if kw.arg}
        missing = REQUIRED_KWARGS - supplied
        if missing:
            problems.append(
                f"{rel}:{node.lineno}: {TARGET} missing required "
                f"{', '.join(sorted(missing))} -- this raises inside the heartbeat "
                f"loop's except block, so the service publishes no heartbeat while "
                f"still looking alive"
            )
    return problems


def main() -> int:
    problems: list[str] = []
    sites = 0
    for path in _iter_python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if TARGET not in text:
            continue
        found = _check_file(path)
        sites += text.count(f"{TARGET}(")
        problems.extend(found)

    if problems:
        print(f"system_health producer gate FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"system_health producer gate OK ({sites} construction site(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
