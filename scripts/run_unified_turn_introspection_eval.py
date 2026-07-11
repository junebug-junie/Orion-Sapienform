#!/usr/bin/env python3
"""Run the unified-turn introspection eval (semantic self-indexing experiment).

Sends each fixture question through the FCC motor path (default_fcc_runner —
the same env-driven `claude -p` seam the harness governor uses), scores the
answer against the ground-truth assertions, and appends one JSONL record per
run with correctness + navigation + context metrics.

The runner does not toggle experiment conditions itself: the operator exports
HARNESS_FCC_MCP_ENABLED plus HARNESS_FCC_GITNEXUS_ENABLED /
HARNESS_FCC_CONTEXT_MODE_ENABLED in THIS process's environment (the FCC turn
runs in-process via default_fcc_runner — no governor restart is involved) and
passes --condition so each record is labeled. The runner refuses to start when
the env flags contradict the claimed condition, so a mislabeled condition can
never silently degrade to baseline. Condition D additionally needs
HARNESS_FCC_CONTEXT_MODE_DIR pointing at an absolute, writable path on the
machine running this script (the .env default /var/lib/orion/context-mode is a
container path; use e.g. /tmp/orion-context-mode on the host). Repo SHA,
GitNexus version, and index staleness are recorded so conditions can be
compared at the same commit.

Usage (host, with FCC server reachable and ~/.fcc/.env present):

    python3 scripts/run_unified_turn_introspection_eval.py \
        --condition A --runs 5 \
        --out /tmp/unified-turn-introspection/condition_a.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from orion.harness.evals.introspection_scoring import (  # noqa: E402
    extract_tool_metrics,
    load_fixture,
    score_answer,
)

CONDITIONS = {
    "A": "baseline",
    "B": "graph_only",
    "C": "graph_plus_breadcrumbs",
    "D": "graph_breadcrumbs_context_mode",
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _tool_version(command: str, *args: str) -> str | None:
    if shutil.which(command) is None:
        return None
    try:
        out = subprocess.check_output(
            [command, *args], text=True, timeout=30, stderr=subprocess.DEVNULL
        ).strip()
        return out.splitlines()[0] if out else None
    except (subprocess.SubprocessError, OSError):
        return None


def _index_state(repo_sha: str) -> dict:
    """Best-effort GitNexus index freshness from the on-disk metadata.

    The index is derived cache; a missing or non-matching commit is reported,
    never silently treated as fresh.
    """
    meta_dir = REPO_ROOT / ".gitnexus"
    if not meta_dir.is_dir():
        return {"index_present": False, "index_stale": None, "index_commit": None}
    indexed_commit = None
    for name in ("gitnexus.json", "meta.json"):
        candidate = meta_dir / name
        if not candidate.is_file():
            continue
        try:
            meta = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        for key in ("commit", "commitSha", "commit_sha", "sha", "headCommit"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                indexed_commit = value
                break
        if indexed_commit:
            break
    stale = None if indexed_commit is None else (indexed_commit != repo_sha)
    return {"index_present": True, "index_stale": stale, "index_commit": indexed_commit}


def _env_truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}


def _check_condition_env(condition: str) -> list[str]:
    """The env flags must match the claimed condition, or every record lies."""
    mcp = _env_truthy("HARNESS_FCC_MCP_ENABLED")
    gitnexus = _env_truthy("HARNESS_FCC_GITNEXUS_ENABLED")
    context_mode = _env_truthy("HARNESS_FCC_CONTEXT_MODE_ENABLED")
    problems: list[str] = []
    if condition == "A":
        if gitnexus:
            problems.append("condition A requires HARNESS_FCC_GITNEXUS_ENABLED off")
        if context_mode:
            problems.append("condition A requires HARNESS_FCC_CONTEXT_MODE_ENABLED off")
    else:
        if not mcp:
            problems.append(f"condition {condition} requires HARNESS_FCC_MCP_ENABLED=true (master flag)")
        if not gitnexus:
            problems.append(f"condition {condition} requires HARNESS_FCC_GITNEXUS_ENABLED=true")
        if condition == "D" and not context_mode:
            problems.append("condition D requires HARNESS_FCC_CONTEXT_MODE_ENABLED=true")
        if condition in ("B", "C") and context_mode:
            problems.append(f"condition {condition} requires HARNESS_FCC_CONTEXT_MODE_ENABLED off")
    return problems


async def _run_question(prompt: str, correlation_id: str, timeout_sec: float) -> dict:
    from orion.harness.runner import default_fcc_runner

    steps: list[dict] = []
    answer = ""
    error = None
    started = time.monotonic()
    async for event in default_fcc_runner(
        prompt=prompt, correlation_id=correlation_id, timeout_sec=timeout_sec
    ):
        etype = event.get("type")
        if etype == "step":
            steps.append(event)
        elif etype == "final":
            answer = str(event.get("llm_response") or "")
        elif etype == "error":
            error = f"{event.get('error_code')}: {event.get('error')}"
            answer = str(event.get("llm_response") or "")
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {"answer": answer, "steps": steps, "error": error, "elapsed_ms": elapsed_ms}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--runs", type=int, default=1, help="repetitions per question")
    parser.add_argument("--question", default=None, help="run only this question id")
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/unified-turn-introspection/results.jsonl"),
    )
    args = parser.parse_args()

    fixture = load_fixture()
    assertions = fixture["assertions"]
    questions = [
        q for q in fixture["questions"] if args.question in (None, q["id"])
    ]
    if not questions:
        print(f"No question matching {args.question!r}", file=sys.stderr)
        return 2

    problems = _check_condition_env(args.condition)
    if problems:
        for problem in problems:
            print(f"flag mismatch: {problem}", file=sys.stderr)
        return 2

    repo_sha = _git_sha()
    static_fields = {
        "condition": CONDITIONS[args.condition],
        "condition_code": args.condition,
        "repo_sha": repo_sha,
        "gitnexus_version": _tool_version("gitnexus", "--version"),
        "context_mode_version": _tool_version("context-mode", "--version"),
        "mcp_master_flag": os.environ.get("HARNESS_FCC_MCP_ENABLED", ""),
        "gitnexus_flag": os.environ.get("HARNESS_FCC_GITNEXUS_ENABLED", ""),
        "context_mode_flag": os.environ.get("HARNESS_FCC_CONTEXT_MODE_ENABLED", ""),
        **_index_state(repo_sha),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("a", encoding="utf-8") as sink:
        for question in questions:
            for run_idx in range(args.runs):
                corr = f"introspect-{args.condition.lower()}-{question['id']}-{run_idx}"
                print(f"[{corr}] running...", flush=True)
                outcome = await _run_question(question["prompt"], corr, args.timeout_sec)
                passed, failed = score_answer(
                    outcome["answer"], question["assertions"], assertions
                )
                metrics = extract_tool_metrics(outcome["steps"])
                record = {
                    "question_id": question["id"],
                    "run_index": run_idx,
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    **static_fields,
                    "answer": outcome["answer"],
                    "error": outcome["error"],
                    "assertions_passed": passed,
                    "assertions_failed": failed,
                    "tool_calls": metrics.tool_calls,
                    "tool_names": metrics.tool_names,
                    "unique_files_read": metrics.unique_files_read,
                    "observed_chars": metrics.observed_chars,
                    "max_context_fill_pct": metrics.max_context_fill_pct,
                    "truncated_results": metrics.truncated_results,
                    "elapsed_ms": outcome["elapsed_ms"],
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                total += 1
                print(
                    f"[{corr}] passed={len(passed)}/{len(question['assertions'])} "
                    f"tools={metrics.tool_calls} elapsed={outcome['elapsed_ms']}ms",
                    flush=True,
                )
    print(f"Wrote {total} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
