"""Regression: the brief Orion reads names why a hire got no help.

2026-10-10: a bare `claude_budget_unobserved` hid the real cause and Orion
built a "regime break" theory around it. The reason must start with a plain
sentence, then a `[code]` block. Since #3397 Claude is the only peer, so the
chain has one link and never mentions Cursor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.claude_peer import describe_claude_refusal
from app.worker import handle_help_request
from orion.curiosity.peer_briefs import format_soft_nudge
from orion.curiosity.role_teach_disclosure import format_budget_spent_progress
from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1


@dataclass
class _ClaudeLimit:
    observed: bool = True
    state: str = "limited"
    staleness_sec: float | None = 1.0


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


def _run(*, claude_limit: Any = None, claude: Any = None) -> PeerBriefV1:
    persisted: list[PeerBriefV1] = []
    calls = {"claude": 0}

    def _claude(*a: Any, **k: Any) -> Any:
        calls["claude"] += 1
        if claude is None:
            raise AssertionError("claude must not spend")
        return claude(*a, **k)

    brief = handle_help_request(
        _help(),
        claude=_claude,
        observe_claude_limit=lambda: claude_limit or _ClaudeLimit(),
        persist=persisted.append,
    )
    assert persisted == [brief]
    if claude is None:
        assert calls["claude"] == 0, "fail-closed behaviour must not change"
    return brief


def test_limited_meter_names_the_cause_without_cursor() -> None:
    brief = _run()
    assert brief.status == "refused_budget"
    assert brief.peer == "claude_room"
    assert brief.refusal_reason == (
        "Claude helper refused: Claude's usage meter reads limited. "
        "[claude_budget_limited]"
    )
    assert "cursor" not in brief.refusal_reason.lower()


def test_describe_claude_refusal_known_and_unknown_codes() -> None:
    assert "spend blind" in describe_claude_refusal("budget_unobserved")
    assert "budget gate returned weird" in describe_claude_refusal("weird")


def test_orion_facing_surfaces_show_the_reason() -> None:
    """The kickoff nudge and the role-teach line are what Orion reads."""
    brief = _run()
    nudge = "\n".join(format_soft_nudge([brief]))
    assert "Claude helper refused" in nudge
    assert "must" not in nudge.lower()

    line = "\n".join(
        format_budget_spent_progress(status="refused_budget", next_hop_n=2, reason=brief.refusal_reason)
    )
    assert "Claude helper refused" in line
    assert "[claude_budget_limited]" not in line  # codes stay out of the short line
    assert "hop 2" in line


def test_long_claude_exception_keeps_codes_inside_nudge_clip() -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("Traceback " + "x" * 5000 + ".")

    brief = _run(
        claude_limit=_ClaudeLimit(observed=True, state="clear", staleness_sec=1.0),
        claude=boom,
    )
    nudge = "\n".join(format_soft_nudge([brief]))
    assert "claude_failed]" in nudge


def test_role_teach_bare_code_keeps_plain_sentence() -> None:
    for bare in ("budget_limited", "claude_budget_unobserved"):
        line = "\n".join(
            format_budget_spent_progress(status="refused_budget", next_hop_n=1, reason=bare)
        )
        assert "Peer hire budget is spent." in line
        assert "Cursor" not in line
        assert bare not in line


def test_old_cursor_rows_still_render() -> None:
    """Rows written before #3397 carry peer=cursor_auto; they stay readable."""
    other = PeerBriefV1(
        brief_id="b-other", help_id="h", run_id="abcd1234abcd", peer="cursor_auto",
        status="failed", summary="", refusal_reason="cursor_other: " + "y" * 2000,
    )
    line = [ln for ln in format_soft_nudge([other]) if "b-other" in ln][0]
    assert len(line.split("Why: ", 1)[1]) <= 300
