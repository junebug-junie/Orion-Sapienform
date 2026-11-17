#!/usr/bin/env python3
"""Eval: contractor peer discrimination — Claude is the only peer (#3397).

Fixture-driven (no live Claude / graph). Measures whether
`handle_help_request` separates:

  - unobserved Claude meter → refused_budget, Claude never called, and the
    reason (as rendered in Orion's kickoff nudge) names the cause plainly
  - limited Claude meter → refused_budget, Claude never called
  - clear meter → Claude asked exactly once, ok brief (conversation-only,
    no evidence claims)
  - clear meter + Claude error → failed brief naming `claude_failed`
  - no Cursor anywhere → nothing mentions Cursor (2026-11-14..16 regression:
    every hire was refused "Cursor unavailable: it is not logged in")

Deterministic, no Docker, no network.

Run:
  PYTHONPATH=services/orion-curiosity-peer:. \\
    python services/orion-curiosity-peer/evals/run_contractor_peer_eval.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_SERVICE_DIR = str(Path(__file__).resolve().parents[1])
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
for _path in (_SERVICE_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.worker import handle_help_request  # noqa: E402
from orion.curiosity.peer_briefs import format_soft_nudge  # noqa: E402
from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1  # noqa: E402


@dataclass
class _Meter:
    observed: bool = True
    state: str = "clear"
    staleness_sec: float | None = 1.0


def _help() -> HelpRequestV1:
    return HelpRequestV1(
        help_id="help-eval-1",
        run_id="run-eval-1",
        prior_id="prior-eval-1",
        mode="world_curiosity",
        question="Why would a retry loop double-count a failure?",
        tried_summary="Read the retry docs.",
        success_criteria="A named mechanism.",
    )


Case = tuple[str, PeerBriefV1, dict[str, int]]
CaseFn = Callable[[], Case]


def _run(name: str, meter: _Meter, reply: Callable[[], Any]) -> Case:
    calls = {"claude": 0}

    def claude(*_a: Any, **_k: Any) -> Any:
        calls["claude"] += 1
        return reply()

    brief = handle_help_request(
        _help(),
        claude=claude,
        observe_claude_limit=lambda: meter,
        persist=lambda _b: None,
    )
    return name, brief, calls


def _never() -> Any:
    raise AssertionError("claude must not spend")


def _boom() -> Any:
    raise TimeoutError("claude utterance timeout after 420.0s")


def _ok() -> str:
    return '{"summary": "from claude once", "evidence_pointers": ["worker.py"]}'


CASES: list[CaseFn] = [
    lambda: _run("unobserved_refuse", _Meter(observed=False, state="unknown", staleness_sec=None), _never),
    lambda: _run("limited_refuse", _Meter(state="limited"), _never),
    lambda: _run("hire_claude", _Meter(), _ok),
    lambda: _run("claude_failed", _Meter(), _boom),
]


def _expect(name: str, brief: PeerBriefV1, calls: dict[str, int]) -> list[str]:
    """Return list of failure strings (empty = pass)."""
    fails: list[str] = []
    seen = "\n".join(format_soft_nudge([brief]))
    if brief.peer != "claude_room":
        fails.append(f"peer={brief.peer} want claude_room")
    if "cursor" in (brief.refusal_reason or "").lower() or "cursor" in seen.lower():
        fails.append("Cursor is retired; nothing Orion reads may mention it")
    if name in ("unobserved_refuse", "limited_refuse"):
        if brief.status != "refused_budget":
            fails.append(f"status={brief.status!r} want refused_budget")
        if calls != {"claude": 0}:
            fails.append(f"calls={calls} want claude=0 (fail-closed)")
        code = "claude_budget_unobserved" if name == "unobserved_refuse" else "claude_budget_limited"
        for needle in ("Claude helper refused", code):
            if needle not in seen:
                fails.append(f"Orion's nudge missing {needle!r}")
    elif name == "hire_claude":
        if brief.status != "ok":
            fails.append(f"status={brief.status} want ok")
        if calls != {"claude": 1}:
            fails.append(f"calls={calls} want claude=1")
        if "from claude once" not in brief.summary:
            fails.append("missing claude summary")
        if brief.evidence_pointers:
            fails.append("claude_room must not claim evidence_pointers")
        if "conversation-only" not in brief.summary.lower():
            fails.append("claude_room summary must be labeled conversation-only")
    elif name == "claude_failed":
        if brief.status != "failed":
            fails.append(f"status={brief.status} want failed")
        if calls != {"claude": 1}:
            fails.append(f"calls={calls} want claude=1 (exactly one try)")
        if "claude_failed" not in seen:
            fails.append("Orion's nudge missing 'claude_failed'")
    else:
        fails.append(f"unknown case {name}")
    return fails


def main() -> int:
    passed = 0
    failed = 0
    for case in CASES:
        name, brief, calls = case()
        fails = _expect(name, brief, calls)
        if fails:
            failed += 1
            print(f"FAIL {name}: {'; '.join(fails)}")
        else:
            passed += 1
            print(f"PASS {name}: status={brief.status} peer={brief.peer} calls={calls}")
    total = passed + failed
    print(f"\ncontractor_peer_eval: {passed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
