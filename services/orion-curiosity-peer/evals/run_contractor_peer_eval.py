#!/usr/bin/env python3
"""Eval: contractor peer discrimination — Claude is the only peer (#3397).

Fixture-driven (no live Claude / graph). Measures whether
`handle_help_request` separates:

  - unobserved Claude meter (idle account) → Claude asked once (#3655;
    11-19..23 live: 13 hires refused while nobody used Claude)
  - limited Claude meter → refused_budget, Claude never called, and the
    reason (as rendered in Orion's kickoff nudge) names the cause plainly
  - Claude answered "you've hit your session limit · resets ..." → the next
    hire is refused_budget naming the reset, Claude never called (#3655;
    11-24 live: the 11:23Z hire repeated the 09:12Z limit failure)
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

from datetime import datetime, timedelta, timezone  # noqa: E402

from app.claude_limit_backoff import ClaudeLimitMemory  # noqa: E402
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


def _run(
    name: str,
    meter: _Meter,
    reply: Callable[[], Any],
    *,
    memory: ClaudeLimitMemory | None = None,
    at: datetime | None = None,
) -> Case:
    calls = {"claude": 0}

    def claude(*_a: Any, **_k: Any) -> Any:
        calls["claude"] += 1
        return reply()

    when = at or datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)
    brief = handle_help_request(
        _help(),
        claude=claude,
        observe_claude_limit=lambda: meter,
        persist=lambda _b: None,
        limit_memory=memory if memory is not None else ClaudeLimitMemory(),
        now=lambda: when,
    )
    return name, brief, calls


_LIMIT_TEXT = "You've hit your session limit · resets 12:10pm (UTC)"


def _limit() -> Any:
    raise RuntimeError(_LIMIT_TEXT)


def _limit_then_hire() -> Case:
    """09:12Z Claude says limit; 11:23Z hire must back off, no call."""
    memory = ClaudeLimitMemory()
    _run("prime", _Meter(), _limit, memory=memory,
         at=datetime(2026, 11, 24, 9, 12, tzinfo=timezone.utc))
    return _run("limit_backoff", _Meter(), _never, memory=memory,
                at=datetime(2026, 11, 24, 11, 23, tzinfo=timezone.utc))


def _after_reset() -> Case:
    memory = ClaudeLimitMemory()
    _run("prime", _Meter(), _limit, memory=memory,
         at=datetime(2026, 11, 24, 9, 12, tzinfo=timezone.utc))
    return _run("after_reset", _Meter(), _ok, memory=memory,
                at=datetime(2026, 11, 24, 12, 10, tzinfo=timezone.utc) + timedelta(seconds=1))


def _never() -> Any:
    raise AssertionError("claude must not spend")


def _boom() -> Any:
    raise TimeoutError("claude utterance timeout after 420.0s")


def _ok() -> str:
    return '{"summary": "from claude once", "evidence_pointers": ["worker.py"]}'


CASES: list[CaseFn] = [
    lambda: _run("unobserved_tries", _Meter(observed=False, state="unknown", staleness_sec=None), _ok),
    lambda: _run("limited_refuse", _Meter(state="limited"), _never),
    lambda: _run("hire_claude", _Meter(), _ok),
    lambda: _run("claude_failed", _Meter(), _boom),
    _limit_then_hire,
    _after_reset,
]


def _expect(name: str, brief: PeerBriefV1, calls: dict[str, int]) -> list[str]:
    """Return list of failure strings (empty = pass)."""
    fails: list[str] = []
    seen = "\n".join(format_soft_nudge([brief]))
    if brief.peer != "claude_room":
        fails.append(f"peer={brief.peer} want claude_room")
    if "cursor" in (brief.refusal_reason or "").lower() or "cursor" in seen.lower():
        fails.append("Cursor is retired; nothing Orion reads may mention it")
    if name in ("limited_refuse", "limit_backoff"):
        if brief.status != "refused_budget":
            fails.append(f"status={brief.status!r} want refused_budget")
        if calls != {"claude": 0}:
            fails.append(f"calls={calls} want claude=0")
        code = "claude_budget_limited" if name == "limited_refuse" else "claude_limit_backoff"
        needles = ["Claude helper refused", code]
        if name == "limit_backoff":
            needles.append("12:10Z")
        for needle in needles:
            if needle not in seen:
                fails.append(f"Orion's nudge missing {needle!r}")
    elif name in ("unobserved_tries", "after_reset"):
        if brief.status != "ok":
            fails.append(f"status={brief.status} want ok")
        if calls != {"claude": 1}:
            fails.append(f"calls={calls} want claude=1")
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
