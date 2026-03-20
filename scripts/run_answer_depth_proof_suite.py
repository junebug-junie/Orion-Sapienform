#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


PASS2_CORE_TESTS = [
    # Cognition-level wiring (no jinja import dependency)
    REPO_ROOT / "orion" / "cognition" / "tests" / "test_runtime_pack_merge.py",
    REPO_ROOT / "orion" / "cognition" / "tests" / "test_live_tool_resolution_pass2.py",
    REPO_ROOT / "orion" / "cognition" / "tests" / "test_agent_chain_guards.py",
    REPO_ROOT / "orion" / "cognition" / "tests" / "test_finalize_payload.py",
    REPO_ROOT / "orion" / "cognition" / "tests" / "test_quality_evaluator.py",
    REPO_ROOT / "tests" / "test_answer_depth_pass2_wiring.py",
]

PASS2_AGENT_CHAIN_TESTS = [
    # Agent-chain runtime guards (import local `app` package)
    REPO_ROOT / "services" / "orion-agent-chain" / "tests" / "test_delivery_verbs.py",
    REPO_ROOT / "services" / "orion-agent-chain" / "tests" / "test_triage_gating.py",
    REPO_ROOT / "services" / "orion-agent-chain" / "tests" / "test_step_cap_finalization.py",
    REPO_ROOT / "services" / "orion-agent-chain" / "tests" / "test_agent_chain_delegate_loop.py",
    REPO_ROOT / "services" / "orion-agent-chain" / "tests" / "test_meta_plan_rewrite_pass2.py",
    REPO_ROOT / "services" / "orion-agent-chain" / "tests" / "test_repeated_plan_action_pass2.py",
]

PASS3_TESTS = [
    REPO_ROOT / "tests" / "test_answer_depth_pass3_golden_path_discord.py",
    REPO_ROOT / "tests" / "test_answer_depth_pass3_supervisor_meta_plan_proof.py",
]


def _has_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def _ensure_test_runtime_deps(*, auto_install: bool) -> None:
    missing: list[str] = []
    if not _has_import("jinja2"):
        missing.append("jinja2")
    if not _has_import("pydantic_settings"):
        missing.append("pydantic-settings")

    if not missing:
        return

    if not auto_install:
        raise SystemExit(
            "Missing dependencies for answer-depth proof suite: "
            + ", ".join(missing)
            + "\nRun: pip install -e \".[test-runtime]\""
        )

    print(f"Installing test-runtime deps: {', '.join(missing)}", file=sys.stderr)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".[test-runtime]"],
        cwd=str(REPO_ROOT),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run answer-depth proof suite (Pass2 + Pass3).")
    parser.add_argument("--no-write", action="store_true", help="Do not write proof artifacts (ORION_PROOF_WRITE=0).")
    parser.add_argument("--auto-install-deps", action="store_true", help="Auto-install .[test-runtime] if missing deps.")
    args = parser.parse_args()

    _ensure_test_runtime_deps(auto_install=args.auto_install_deps)

    env = os.environ.copy()
    if args.no_write:
        env["ORION_PROOF_WRITE"] = "0"
    else:
        env["ORION_PROOF_WRITE"] = env.get("ORION_PROOF_WRITE") or "1"

    def _run(test_paths: list[Path], *, python_path: list[Path]) -> None:
        env_run = env.copy()
        env_run["PYTHONPATH"] = ":".join([str(p) for p in python_path])
        cmd = [sys.executable, "-m", "pytest", "-q", *[str(p) for p in test_paths]]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env_run)

    # Critical: avoid `app` package collisions by isolating service roots.
    # - agent-chain tests require `services/orion-agent-chain` to be on PYTHONPATH, but not cortex-exec.
    # - pass3 tests dynamically import both services and clear `sys.modules['app']`, so they can run with repo root only.
    _run(PASS2_CORE_TESTS, python_path=[REPO_ROOT])
    _run(PASS2_AGENT_CHAIN_TESTS, python_path=[REPO_ROOT, REPO_ROOT / "services" / "orion-agent-chain"])
    _run(PASS3_TESTS, python_path=[REPO_ROOT])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

