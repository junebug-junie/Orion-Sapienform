#!/usr/bin/env python3
"""Compact CLI runner for context-exec RLM eval fixtures."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _script_dir:
    sys.path.pop(0)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SERVICE_DIR = os.path.join(REPO_ROOT, "services", "orion-context-exec")
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

from orion.schemas.context_exec import ContextExecPermissionV1  # noqa: E402

from app.rlm_eval_harness import (  # noqa: E402
    ALL_EVAL_CASES,
    REPO_ROOT,
    run_eval_case,
    seed_case_organs,
)


def _quality_pass(case_name: str, engine: str, run) -> bool:
    if run.status != "ok":
        return False
    if case_name == "belief_provenance_denver" and engine == "alexzhang":
        return False
    return True


async def _run(engine: str) -> list[tuple[str, str, str, str, str, bool, str]]:
    rows: list[tuple[str, str, str, str, str, bool, str]] = []
    repo_settings = {
        "context_exec_repo_root": REPO_ROOT,
        "orion_repo_root": REPO_ROOT,
        "context_exec_real_repo_enabled": True,
    }
    for case in ALL_EVAL_CASES:
        seed_case_organs(case.name)
        notes = ""
        perms = (
            ContextExecPermissionV1(read_repo=True)
            if case.mode == "repo_impact_analysis"
            else None
        )
        overrides = repo_settings if case.mode == "repo_impact_analysis" else None
        try:
            run = await run_eval_case(
                engine,
                case,
                permissions=perms,
                settings_overrides=overrides,
            )
            artifact_status = str((run.artifact or {}).get("status", run.status))
            quality = _quality_pass(case.name, engine, run)
        except Exception as exc:
            artifact_status = "error"
            quality = False
            notes = str(exc)
            run = None
        if run is not None:
            rows.append(
                (
                    case.name,
                    engine,
                    case.mode,
                    artifact_status,
                    str(run.artifact_type or ""),
                    quality,
                    notes,
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run context-exec RLM eval fixtures")
    parser.add_argument("--engine", choices=["fake", "alexzhang"], default="fake")
    args = parser.parse_args()

    rows = asyncio.run(_run(args.engine))
    header = ("case", "engine", "mode", "status", "artifact_type", "quality_pass", "notes")
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(str(c) for c in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
