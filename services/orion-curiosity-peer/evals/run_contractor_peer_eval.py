#!/usr/bin/env python3
"""Eval: contractor peer discrimination — budget refuse vs hire; fallback once.

Fixture-driven (no live Cursor / Claude / graph). Measures whether
`handle_help_request` separates:

  - contested / unobserved budget → refused_budget, peers never called
  - clear budget + Cursor ok → hire (cursor_auto), Claude never called
  - clear budget + Cursor token_unavailable → Claude fallback exactly once

Deterministic, no Docker, no network. Matches the shape of
services/orion-exo-exploration/evals/run_interest_scoring_eval.py.

Run:
  PYTHONPATH=services/orion-curiosity-peer:. \\
    python services/orion-curiosity-peer/evals/run_contractor_peer_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

_SERVICE_DIR = str(Path(__file__).resolve().parents[1])
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
for _path in (_SERVICE_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.cursor_errors import TokenUnavailable  # noqa: E402
from app.worker import handle_help_request  # noqa: E402
from orion.dev_economics.cursor_limit_events import CursorLimitObservation  # noqa: E402
from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1  # noqa: E402


def _help() -> HelpRequestV1:
    return HelpRequestV1(
        help_id="help-eval-1",
        run_id="run-eval-1",
        prior_id="prior-eval-1",
        mode="world_curiosity",
        question="Where is decide_cursor_budget?",
        tried_summary="Looked at scarcity docs.",
        success_criteria="File path + function name.",
    )


def _clear() -> CursorLimitObservation:
    return CursorLimitObservation(observed=True, state="clear", staleness_sec=1.0)


def _unobserved() -> CursorLimitObservation:
    return CursorLimitObservation(observed=False, state="unknown", staleness_sec=None)


CaseFn = Callable[[], tuple[str, PeerBriefV1, dict[str, int]]]


def _case_budget_refuse() -> tuple[str, PeerBriefV1, dict[str, int]]:
    calls = {"cursor": 0, "claude": 0}

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        raise AssertionError("cursor must not run")

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        raise AssertionError("claude must not run")

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=_unobserved,
        persist=lambda _b: None,
    )
    return "budget_refuse", brief, calls


def _case_hire_cursor() -> tuple[str, PeerBriefV1, dict[str, int]]:
    calls = {"cursor": 0, "claude": 0}

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        return PeerBriefV1(
            brief_id="brief-hire",
            help_id="help-eval-1",
            run_id="run-eval-1",
            prior_id="prior-eval-1",
            peer="cursor_auto",
            status="ok",
            summary="decide_cursor_budget in cursor_limit_events.py",
            evidence_pointers=["orion/dev_economics/cursor_limit_events.py"],
        )

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        return "should not run"

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=_clear,
        persist=lambda _b: None,
    )
    return "hire_cursor", brief, calls


def _case_fallback_once() -> tuple[str, PeerBriefV1, dict[str, int]]:
    from dataclasses import dataclass

    calls = {"cursor": 0, "claude": 0}

    @dataclass
    class _ClearClaude:
        observed: bool = True
        state: str = "clear"
        staleness_sec: float | None = 1.0

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        calls["cursor"] += 1
        raise TokenUnavailable("dry")

    def claude(*_a: Any, **_k: Any) -> str:
        calls["claude"] += 1
        return '{"summary": "from claude once", "evidence_pointers": ["worker.py"]}'

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=claude,
        observe_limit=_clear,
        observe_claude_limit=lambda: _ClearClaude(),
        persist=lambda _b: None,
    )
    return "fallback_once", brief, calls


def _expect(
    name: str,
    brief: PeerBriefV1,
    calls: dict[str, int],
) -> list[str]:
    """Return list of failure strings (empty = pass)."""
    fails: list[str] = []
    if name == "budget_refuse":
        if brief.status != "refused_budget":
            fails.append(f"status={brief.status!r} want refused_budget")
        if calls != {"cursor": 0, "claude": 0}:
            fails.append(f"calls={calls} want both 0")
    elif name == "hire_cursor":
        if brief.status != "ok" or brief.peer != "cursor_auto":
            fails.append(f"status/peer={brief.status}/{brief.peer} want ok/cursor_auto")
        if calls != {"cursor": 1, "claude": 0}:
            fails.append(f"calls={calls} want cursor=1 claude=0")
    elif name == "fallback_once":
        if brief.status != "ok" or brief.peer != "claude_room":
            fails.append(f"status/peer={brief.status}/{brief.peer} want ok/claude_room")
        if calls != {"cursor": 1, "claude": 1}:
            fails.append(f"calls={calls} want cursor=1 claude=1")
        if "from claude once" not in brief.summary:
            fails.append("missing claude summary")
        if brief.evidence_pointers:
            fails.append("claude_room must not claim evidence_pointers")
        if "conversation-only" not in brief.summary.lower():
            fails.append("claude_room summary must be labeled conversation-only")
    else:
        fails.append(f"unknown case {name}")
    return fails


def main() -> int:
    cases: list[CaseFn] = [_case_budget_refuse, _case_hire_cursor, _case_fallback_once]
    passed = 0
    failed = 0
    for case in cases:
        name, brief, calls = case()
        fails = _expect(name, brief, calls)
        if fails:
            failed += 1
            print(f"FAIL {name}: {'; '.join(fails)}")
        else:
            passed += 1
            print(
                f"PASS {name}: status={brief.status} peer={brief.peer} "
                f"calls={calls}"
            )
    total = passed + failed
    print(f"\ncontractor_peer_eval: {passed}/{total} passed")
    # Discrimination floor: all three labeled paths must separate correctly.
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
