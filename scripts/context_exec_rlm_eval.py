#!/usr/bin/env python3
"""Compact CLI runner for context-exec RLM eval fixtures."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SERVICE_DIR = os.path.join(REPO_ROOT, "services", "orion-context-exec")
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

_FIXTURES_PATH = os.path.join(SERVICE_DIR, "tests", "test_rlm_eval_fixtures.py")
_spec = importlib.util.spec_from_file_location("test_rlm_eval_fixtures", _FIXTURES_PATH)
_fixtures = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fixtures)

BELIEF_DENVER = _fixtures.BELIEF_DENVER
BELIEF_OGDEN = _fixtures.BELIEF_OGDEN
REPO_AGENT_CHAIN = _fixtures.REPO_AGENT_CHAIN
REPO_ENGINE_FILES_CASE = _fixtures.REPO_ENGINE_FILES_CASE
TRACE_KNOWN_CORR = _fixtures.TRACE_KNOWN_CORR
TRACE_MISSING_CORR = _fixtures.TRACE_MISSING_CORR
FAKE_ORGANS = _fixtures.FAKE_ORGANS
run_eval_case = _fixtures.run_eval_case

from orion.schemas.context_exec import ContextExecPermissionV1  # noqa: E402

import pytest  # noqa: E402

EVAL_CASES = [
    BELIEF_DENVER,
    BELIEF_OGDEN,
    TRACE_KNOWN_CORR,
    TRACE_MISSING_CORR,
    REPO_AGENT_CHAIN,
    REPO_ENGINE_FILES_CASE,
]


def _quality_pass(case_name: str, engine: str, run) -> bool:
    if run.status != "ok":
        return False
    if case_name == "belief_provenance_denver" and engine == "alexzhang":
        return False
    return True


async def _run(engine: str) -> list[tuple[str, str, str, str, str, bool, str]]:
    rows: list[tuple[str, str, str, str, str, bool, str]] = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("app.runner.settings.context_exec_repo_root", REPO_ROOT)
        monkeypatch.setattr("app.runner.settings.orion_repo_root", REPO_ROOT)
        monkeypatch.setattr("app.runner.settings.context_exec_real_repo_enabled", True)
        for case in EVAL_CASES:
            FAKE_ORGANS.memory_hits = None
            FAKE_ORGANS.trace_hits = None
            notes = ""
            if case.name == "belief_provenance_denver":
                FAKE_ORGANS.memory_hits = [
                    {
                        "claim": "User is from Denver",
                        "source_ref": "m:denver:1",
                        "verified": True,
                        "confidence": 0.9,
                    }
                ]
            elif case.name == "belief_provenance_ogden":
                FAKE_ORGANS.memory_hits = [
                    {
                        "claim": "User location is Ogden, not Denver",
                        "source_ref": "m:ogden:1",
                        "verified": True,
                        "confidence": 0.9,
                    }
                ]
            elif case.name == "trace_autopsy_known_corr":
                FAKE_ORGANS.trace_hits = [
                    {
                        "handle": "t:5506",
                        "snippet": "fail open fallback to AgentChainService after ContextExecService timeout",
                        "corr_id": "5506f854-2606-42b5-a2ee-89775b3a8ed5",
                        "kind": "error",
                        "source": "cortex",
                    }
                ]
            perms = ContextExecPermissionV1(read_repo=True) if case.mode == "repo_impact_analysis" else None
            try:
                run = await run_eval_case(engine, case, monkeypatch, permissions=perms)
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
    finally:
        monkeypatch.undo()
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
