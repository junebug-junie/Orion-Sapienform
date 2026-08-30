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
# Not schema-required (it has a default of 10.0) but required to be CORRECT.
# orion-equilibrium-service computes grace = heartbeat_interval_sec *
# EQUILIBRIUM_GRACE_MULTIPLIER (3.0) and marks a service "down" once the gap
# exceeds it. Every loop in this repo sleeps 30s, so the 10.0 default yields a
# 30.0s grace against a 30s period -- zero margin, and any event-loop delay or bus
# latency flips the service to "down", emits a spurious transition and pushes
# distress_score. Omitting it is not a style issue, it is a live false alarm.
INTERVAL_KWARG = "heartbeat_interval_sec"
# A gate that inspects nothing must fail, not pass. If SEARCH_ROOTS ever miss --
# renamed dirs, or a checkout path containing a SKIP_PARTS component -- every file
# is skipped and a naive gate prints OK. This repo has been burned by an inert
# gate before, so the floor is asserted rather than assumed.
MIN_EXPECTED_SITES = 8
SEARCH_ROOTS = ("services", "orion")
SKIP_PARTS = {".git", "__pycache__", "node_modules", ".venv", ".worktrees", "tests", "evals"}
# Matched against the path RELATIVE to the repo root, never the absolute path: a
# checkout living under e.g. /home/x/tests/ would otherwise skip the entire repo
# and the gate would pass having read nothing.


def _iter_python_files():
    for root in SEARCH_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if SKIP_PARTS & set(path.relative_to(REPO_ROOT).parts):
                continue
            yield path


def _target_aliases(tree: ast.Module) -> set[str]:
    """Local names bound to SystemHealthV1, including `as` aliases.

    `from ... import SystemHealthV1 as SH` followed by `SH(...)` was invisible to an
    earlier version of this gate: the import line satisfied the text prefilter, so
    the file was scanned and reported clean while the call went unchecked.
    """
    names = {TARGET}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == TARGET and alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith(f".{TARGET}") and alias.asname:
                    names.add(alias.asname)
    return names


def _check_file(path: Path) -> tuple[list[str], int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return ([f"{path.relative_to(REPO_ROOT)}: could not parse ({exc})"], 0)

    aliases = _target_aliases(tree)
    problems: list[str] = []
    call_sites = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        rel = path.relative_to(REPO_ROOT)

        # `model_construct` skips validation entirely, so a missing required field
        # would neither be caught here nor raise at runtime. `model_validate` is
        # deliberately NOT flagged: it fully validates, so it fails loudly on its
        # own, and it is how the legitimate CONSUMER reads heartbeats off the bus
        # (services/orion-equilibrium-service/app/service.py). Flagging it was a
        # false positive in this gate's first cut.
        if name == "model_construct":
            owner = getattr(func, "value", None)
            if getattr(owner, "id", None) in aliases:
                problems.append(
                    f"{rel}:{node.lineno}: {TARGET}.{name}(...) bypasses this gate -- "
                    f"construct it with keyword arguments instead"
                )
            continue

        if name not in aliases:
            continue
        call_sites += 1
        if any(kw.arg is None for kw in node.keywords):
            problems.append(
                f"{rel}:{node.lineno}: {TARGET}(**kwargs) -- cannot verify required "
                f"fields statically; pass them explicitly"
            )
            continue

        supplied = {kw.arg for kw in node.keywords if kw.arg}
        if INTERVAL_KWARG not in supplied:
            problems.append(
                f"{rel}:{node.lineno}: {TARGET} does not pass {INTERVAL_KWARG} -- it "
                f"defaults to 10.0, giving a 30.0s equilibrium grace; a loop slower "
                f"than that is classified \"down\" on every tick"
            )
        missing = REQUIRED_KWARGS - supplied
        if missing:
            problems.append(
                f"{rel}:{node.lineno}: {TARGET} missing required "
                f"{', '.join(sorted(missing))} -- this raises inside the heartbeat "
                f"loop's except block, so the service publishes no heartbeat while "
                f"still looking alive"
            )
    return problems, call_sites


def main() -> int:
    problems: list[str] = []
    sites = 0
    for path in _iter_python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if TARGET not in text:
            continue
        found, call_sites = _check_file(path)
        # AST-verified call count, not a substring count -- the latter also matched
        # `class SystemHealthV1(BaseModel):` and so overstated coverage by one.
        sites += call_sites
        problems.extend(found)

    if sites < MIN_EXPECTED_SITES:
        print(
            f"system_health producer gate FAILED: found only {sites} construction "
            f"site(s), expected at least {MIN_EXPECTED_SITES}. The gate is not "
            f"reaching the code it is meant to check."
        )
        return 1

    if problems:
        print(f"system_health producer gate FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"system_health producer gate OK ({sites} construction site(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
