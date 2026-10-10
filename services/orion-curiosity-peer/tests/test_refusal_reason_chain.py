"""Regression: the brief Orion reads names the whole refusal chain (2026-10-10).

Every HelpRequest since ~09-28 was refused because Cursor hit its monthly
usage limit; the brief said only `claude_budget_unobserved`, and Orion built a
"regime break" theory around a billing cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.cursor_errors import (
    TokenUnavailable,
    cursor_token_cause,
    describe_claude_refusal,
    parse_cursor_reset_date,
)
from app.worker import handle_help_request
from orion.curiosity.peer_briefs import format_soft_nudge
from orion.curiosity.role_teach_disclosure import format_budget_spent_progress
from orion.dev_economics.cursor_limit_events import CursorLimitObservation
from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1

# Verbatim from orion-athena-curiosity-peer logs (2026-10-09 10:04:15 UTC).
REAL_CURSOR_ERROR = (
    "cursor agent exited 1: S: You've hit your usage limit You've saved $1256 on API "
    "model usage this month with Pro+. Switch to a different model or set a Spend "
    "Limit to continue with this model. Your usage limits will reset when your "
    "monthly cycle ends on 10/14/2026."
)


@dataclass
class _ClaudeLimit:
    observed: bool = False
    state: str = "unknown"
    staleness_sec: float | None = None


def _help() -> HelpRequestV1:
    return HelpRequestV1(
        help_id="help-chain",
        run_id="run-chain",
        prior_id="prior-chain",
        mode="world_curiosity",
        question="Why do peer asks keep failing?",
        tried_summary="Counted refusals.",
        success_criteria="A named cause.",
    )


def _run(cursor_exc: BaseException, *, claude_limit: Any = None, claude: Any = None) -> PeerBriefV1:
    persisted: list[PeerBriefV1] = []
    calls = {"claude": 0}

    def cursor(*_a: Any, **_k: Any) -> PeerBriefV1:
        raise cursor_exc

    def _claude(*a: Any, **k: Any) -> Any:
        calls["claude"] += 1
        if claude is None:
            raise AssertionError("claude must not spend")
        return claude(*a, **k)

    brief = handle_help_request(
        _help(),
        cursor=cursor,
        claude=_claude,
        observe_limit=lambda: CursorLimitObservation(observed=True, state="clear", staleness_sec=1.0),
        observe_claude_limit=lambda: claude_limit or _ClaudeLimit(),
        persist=persisted.append,
    )
    assert persisted == [brief]
    if claude is None:
        assert calls["claude"] == 0, "fail-closed behaviour must not change"
    return brief


def test_real_cursor_usage_limit_names_whole_chain() -> None:
    brief = _run(RuntimeError(REAL_CURSOR_ERROR))
    reason = brief.refusal_reason or ""
    assert brief.status == "refused_budget"
    assert brief.peer == "claude_room"
    assert "Cursor unavailable: it hit its usage limit (resets 2026-10-14)." in reason
    assert "Claude fallback refused" in reason
    assert "cannot see Claude's usage meter" in reason
    assert "cursor_token_unavailable:usage_limit" in reason
    assert "claude_budget_unobserved" in reason
    # Cursor's own words survive as evidence.
    assert "monthly cycle ends on 10/14/2026" in reason


def test_usage_limit_without_date_says_unknown() -> None:
    brief = _run(RuntimeError("cursor agent exited 1: You've hit your usage limit."))
    reason = brief.refusal_reason or ""
    assert "usage limit (reset date unknown)" in reason
    assert "2026" not in reason.split("cursor said:")[0]


def test_non_token_cursor_failure_path_unchanged() -> None:
    brief = _run(RuntimeError("segfault in agent"))
    assert brief.status == "failed"
    assert brief.peer == "cursor_auto"
    assert brief.refusal_reason == "cursor_other: segfault in agent"


def test_bare_token_unavailable_does_not_invent_a_cause() -> None:
    brief = _run(TokenUnavailable("dry"))
    reason = brief.refusal_reason or ""
    assert "reason not recognised" in reason
    assert "logged in" not in reason
    assert "usage limit" not in reason


def test_auth_and_binary_causes_carried_through() -> None:
    auth = _run(RuntimeError("cursor agent exited 1: Not logged in. Run `agent login`."))
    assert "not logged in" in (auth.refusal_reason or "").lower()
    assert "cursor_token_unavailable:auth" in (auth.refusal_reason or "")
    missing = _run(TokenUnavailable("cursor agent binary not found: agent"))
    assert "not installed" in (missing.refusal_reason or "")
    assert "cursor_token_unavailable:binary_missing" in (missing.refusal_reason or "")


def test_claude_failed_path_also_names_cursor_cause() -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("room timeout")

    brief = _run(
        RuntimeError(REAL_CURSOR_ERROR),
        claude_limit=_ClaudeLimit(observed=True, state="clear", staleness_sec=1.0),
        claude=boom,
    )
    reason = brief.refusal_reason or ""
    assert brief.status == "failed"
    assert "resets 2026-10-14" in reason
    assert "claude_failed" in reason and "room timeout" in reason


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Your usage limits will reset when your monthly cycle ends on 10/14/2026.", "2026-10-14"),
        ("limits reset on 1/2/2027", "2027-01-02"),
        ("You've hit your usage limit.", None),
        ("ends on 10/14/2026", None),  # no "reset" — do not guess
        ("will reset on 13/40/2026", None),  # impossible date
        ("", None),
    ],
)
def test_parse_cursor_reset_date(text: str, want: str | None) -> None:
    assert parse_cursor_reset_date(text) == want


def test_cause_classification() -> None:
    assert cursor_token_cause(RuntimeError(REAL_CURSOR_ERROR)) == "usage_limit"
    assert cursor_token_cause(RuntimeError("402 payment required")) == "usage_limit"
    assert cursor_token_cause(RuntimeError("401 unauthorized")) == "auth"
    assert cursor_token_cause(TokenUnavailable("")) == "unknown"
    assert "spend blind" in describe_claude_refusal("budget_unobserved")
    assert "budget gate returned weird" in describe_claude_refusal("weird")


def test_orion_facing_surfaces_show_the_reason() -> None:
    """The kickoff nudge and the role-teach line are what Orion reads."""
    brief = _run(RuntimeError(REAL_CURSOR_ERROR))
    nudge = "\n".join(format_soft_nudge([brief]))
    assert "usage limit (resets 2026-10-14)" in nudge
    assert "Claude fallback refused" in nudge
    assert "must" not in nudge.lower()

    line = "\n".join(
        format_budget_spent_progress(status="refused_budget", next_hop_n=2, reason=brief.refusal_reason)
    )
    assert "usage limit (resets 2026-10-14)" in line
    assert "Claude fallback refused" in line
    assert "cursor said" not in line  # codes + raw text stay out of the short line
    assert "hop 2" in line
