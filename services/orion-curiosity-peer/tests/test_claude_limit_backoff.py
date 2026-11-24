"""Regression for #3655: the Claude gate was wrong both ways.

1. 2026-11-19..23: no Claude Code session ran, the transcript meter read
   `unknown`, and 13 hires were refused `claude_budget_unobserved`.
2. 2026-11-17 20:28Z and 2026-11-24 09:12Z/11:23Z: the meter read `clear` and
   Claude answered "You've hit your session limit · resets ...". The second
   11-24 hire repeated the failure the first one had already announced.

Rule (Juniper 2026-11-24): try, back off on refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.claude_limit_backoff import (
    RECENT_CLAUDE_OUTCOME_CYPHER,
    ClaudeLimitMemory,
    parse_limit_reset,
)
from app.settings import Settings
from app.worker import handle_help_request, seed_claude_limit_memory
from orion.schemas.curiosity_peer import HelpRequestV1, PeerBriefV1

LIVE_TEXT_1124 = "You've hit your session limit · resets 12:10pm (UTC)"
LIVE_TEXT_1117 = "You've hit your session limit · resets 11pm (UTC)"


@dataclass
class _Meter:
    observed: bool = True
    state: str = "clear"
    staleness_sec: float | None = 1.0


IDLE = _Meter(observed=False, state="unknown", staleness_sec=None)


def _help(n: int = 1) -> HelpRequestV1:
    return HelpRequestV1(
        help_id=f"help-backoff-{n}",
        run_id=f"run-backoff-{n}",
        prior_id="prior-backoff",
        mode="world_curiosity",
        question="Why does the meter disagree with Claude?",
        tried_summary="Read the meter.",
        success_criteria="A named cause.",
    )


class _Clock:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at


class _Claude:
    def __init__(self) -> None:
        self.calls = 0
        self.fail_with: str | None = None

    def __call__(self, *_a: Any, **_k: Any) -> str:
        self.calls += 1
        if self.fail_with is not None:
            raise RuntimeError(self.fail_with)
        return '{"summary": "an answer"}'


def _hire(n, *, claude, meter, memory, clock, settings=None) -> PeerBriefV1:
    persisted: list[PeerBriefV1] = []
    brief = handle_help_request(
        _help(n),
        settings=settings or Settings(_env_file=None),
        claude=claude,
        observe_claude_limit=lambda: meter,
        persist=persisted.append,
        limit_memory=memory,
        now=clock,
    )
    assert persisted == [brief]
    return brief


def test_idle_meter_calls_claude() -> None:
    """Live failure 1: unobserved meter (idle account) now reaches Claude."""
    claude = _Claude()
    brief = _hire(
        1,
        claude=claude,
        meter=IDLE,
        memory=ClaudeLimitMemory(),
        clock=_Clock(datetime(2026, 11, 21, 11, 28, tzinfo=timezone.utc)),
    )
    assert claude.calls == 1
    assert brief.status == "ok"


def test_transcript_limited_still_refuses_without_a_call() -> None:
    claude = _Claude()
    brief = _hire(
        1,
        claude=claude,
        meter=_Meter(state="limited"),
        memory=ClaudeLimitMemory(),
        clock=_Clock(datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)),
    )
    assert claude.calls == 0
    assert brief.status == "refused_budget"
    assert brief.refusal_reason.endswith("[claude_budget_limited]")


def test_session_limit_failure_backs_off_until_reset_then_calls_again() -> None:
    """Live failure 2 (11-24): 09:12Z failed 'resets 12:10pm (UTC)'; the
    11:23Z hire must be refused naming the reset, with no Claude call; a hire
    after 12:10Z calls Claude again."""
    memory = ClaudeLimitMemory()
    clock = _Clock(datetime(2026, 11, 24, 9, 12, 50, tzinfo=timezone.utc))
    claude = _Claude()
    claude.fail_with = LIVE_TEXT_1124

    first = _hire(1, claude=claude, meter=_Meter(), memory=memory, clock=clock)
    assert claude.calls == 1
    assert first.status == "failed"
    assert first.refusal_reason.endswith("[claude_failed]")

    clock.at = datetime(2026, 11, 24, 11, 23, 54, tzinfo=timezone.utc)
    second = _hire(2, claude=claude, meter=_Meter(), memory=memory, clock=clock)
    assert claude.calls == 1, "no Claude call while backing off"
    assert second.status == "refused_budget"
    assert "2026-11-24 12:10Z" in second.refusal_reason
    assert second.refusal_reason.startswith("Claude helper refused: ")
    assert second.refusal_reason.endswith("[claude_limit_backoff]")

    claude.fail_with = None
    clock.at = datetime(2026, 11, 24, 12, 10, 0, tzinfo=timezone.utc)
    third = _hire(3, claude=claude, meter=_Meter(), memory=memory, clock=clock)
    assert claude.calls == 2
    assert third.status == "ok"


def test_backoff_holds_even_when_the_meter_is_idle() -> None:
    """11-17 shape on an idle meter: unknown must not override a stated limit."""
    memory = ClaudeLimitMemory()
    clock = _Clock(datetime(2026, 11, 17, 20, 28, 37, tzinfo=timezone.utc))
    claude = _Claude()
    claude.fail_with = LIVE_TEXT_1117
    _hire(1, claude=claude, meter=IDLE, memory=memory, clock=clock)
    clock.at = datetime(2026, 11, 17, 22, 59, tzinfo=timezone.utc)
    brief = _hire(2, claude=claude, meter=IDLE, memory=memory, clock=clock)
    assert claude.calls == 1
    assert brief.status == "refused_budget"
    assert "2026-11-17 23:00Z" in brief.refusal_reason


def test_unparseable_reset_uses_bounded_default() -> None:
    memory = ClaudeLimitMemory()
    t0 = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)
    clock = _Clock(t0)
    claude = _Claude()
    claude.fail_with = "You've hit your usage limit"
    settings = Settings(_env_file=None, CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC=1800)
    _hire(1, claude=claude, meter=_Meter(), memory=memory, clock=clock, settings=settings)

    clock.at = t0 + timedelta(minutes=29)
    held = _hire(2, claude=claude, meter=_Meter(), memory=memory, clock=clock, settings=settings)
    assert claude.calls == 1
    assert held.status == "refused_budget"
    assert "without a reset time" in held.refusal_reason
    assert "09:30Z" in held.refusal_reason

    claude.fail_with = None
    clock.at = t0 + timedelta(minutes=30)
    _hire(3, claude=claude, meter=_Meter(), memory=memory, clock=clock, settings=settings)
    assert claude.calls == 2


def test_non_limit_failure_does_not_back_off() -> None:
    memory = ClaudeLimitMemory()
    clock = _Clock(datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc))
    claude = _Claude()
    claude.fail_with = "claude utterance timeout after 420.0s"
    _hire(1, claude=claude, meter=_Meter(), memory=memory, clock=clock)
    clock.at += timedelta(minutes=1)
    _hire(2, claude=claude, meter=_Meter(), memory=memory, clock=clock)
    assert claude.calls == 2


def test_default_backoff_setting_is_one_hour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC", raising=False)
    assert Settings(_env_file=None).CURIOSITY_PEER_CLAUDE_LIMIT_BACKOFF_DEFAULT_SEC == 3600.0


@pytest.mark.parametrize(
    ("text", "at", "want"),
    [
        (LIVE_TEXT_1124, datetime(2026, 11, 24, 9, 12, tzinfo=timezone.utc),
         datetime(2026, 11, 24, 12, 10, tzinfo=timezone.utc)),
        (LIVE_TEXT_1117, datetime(2026, 11, 17, 20, 28, tzinfo=timezone.utc),
         datetime(2026, 11, 17, 23, 0, tzinfo=timezone.utc)),
        # Past today -> tomorrow.
        ("You've hit your session limit · resets 3am (UTC)",
         datetime(2026, 11, 24, 23, 0, tzinfo=timezone.utc),
         datetime(2026, 11, 25, 3, 0, tzinfo=timezone.utc)),
        # A named zone: 11pm Denver (MST, UTC-7 in November) = 06:00Z next day.
        ("You've hit your session limit · resets 11pm (America/Denver)",
         datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc),
         datetime(2026, 11, 25, 6, 0, tzinfo=timezone.utc)),
        # No zone given -> UTC, as Claude's companion runs.
        ("You've hit your session limit · resets 11pm",
         datetime(2026, 11, 17, 20, 28, tzinfo=timezone.utc),
         datetime(2026, 11, 17, 23, 0, tzinfo=timezone.utc)),
        ("You've hit your weekly limit · resets Nov 27, 3am (UTC)",
         datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc),
         datetime(2026, 11, 27, 3, 0, tzinfo=timezone.utc)),
        ("You've hit your usage limit", datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc), None),
        # Not an IANA zone: no guess (the caller's bounded default applies).
        ("You've hit your session limit · resets 11pm (PST)",
         datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc), None),
    ],
)
def test_parse_limit_reset(text: str, at: datetime, want: datetime | None) -> None:
    assert parse_limit_reset(text, at) == want


class _Graph:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def graph_query(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append((cypher, params))
        return self.rows


def _ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


def test_boot_seed_from_recent_limit_failure_refuses_next_hire() -> None:
    """A restart between 09:12Z and 12:10Z must not forget the limit."""
    failed_at = datetime(2026, 11, 24, 9, 12, 50, tzinfo=timezone.utc)
    boot = datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc)
    graph = _Graph([
        {"written_at": _ms(failed_at), "status": "failed",
         "refusal_reason": f"Claude helper was tried and failed: {LIVE_TEXT_1124}. [claude_failed]",
         "summary": ""},
    ])
    memory = ClaudeLimitMemory()
    hold = seed_claude_limit_memory(
        Settings(_env_file=None), graph_client=graph, memory=memory, now=boot
    )
    assert hold is not None
    assert graph.calls[0][0] == RECENT_CLAUDE_OUTCOME_CYPHER
    assert graph.calls[0][1]["since_ms"] == _ms(boot - timedelta(hours=5))

    claude = _Claude()
    brief = _hire(1, claude=claude, meter=_Meter(), memory=memory, clock=_Clock(boot))
    assert claude.calls == 0
    assert brief.status == "refused_budget"
    assert "12:10Z" in brief.refusal_reason


def test_boot_seed_ignores_old_or_ok_rows() -> None:
    boot = datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)
    memory = ClaudeLimitMemory()
    old = {"written_at": _ms(boot - timedelta(hours=6)), "status": "failed",
           "refusal_reason": LIVE_TEXT_1124, "summary": ""}
    ok = {"written_at": _ms(boot - timedelta(hours=1)), "status": "ok",
          "refusal_reason": None, "summary": "fine"}
    assert seed_claude_limit_memory(
        Settings(_env_file=None), graph_client=_Graph([old]), memory=memory, now=boot
    ) is None
    assert seed_claude_limit_memory(
        Settings(_env_file=None), graph_client=_Graph([ok]), memory=memory, now=boot
    ) is None
    assert memory.active(boot) is None


def test_boot_seed_graph_error_is_best_effort() -> None:
    class _Broken:
        def graph_query(self, *_a: Any, **_k: Any) -> Any:
            raise ConnectionError("falkor down")

    assert seed_claude_limit_memory(
        Settings(_env_file=None), graph_client=_Broken(), memory=ClaudeLimitMemory(),
        now=datetime(2026, 11, 24, tzinfo=timezone.utc),
    ) is None


def test_boot_seed_skips_without_graph_credentials() -> None:
    settings = Settings(_env_file=None, ORION_CURIOSITY_GRAPH_HOST="", ORION_CURIOSITY_GRAPH_PORT="")
    assert seed_claude_limit_memory(settings, memory=ClaudeLimitMemory()) is None


def test_reset_read_just_after_it_passed_does_not_hold_a_day() -> None:
    """Review finding: 'resets 12:10pm' read at 12:10:20 rolls to tomorrow in
    parse_reset_at. A dateless session reset is never > 5 h away, so use the
    bounded default instead of a ~24 h hold."""
    from app.claude_limit_backoff import hold_from_failure

    at = datetime(2026, 11, 24, 12, 10, 20, tzinfo=timezone.utc)
    hold = hold_from_failure(LIVE_TEXT_1124, at, default_sec=3600)
    assert hold is not None
    assert hold.resets_at == at + timedelta(hours=1)
    assert hold.reset_stated is False


def test_weekly_reset_days_away_is_kept() -> None:
    from app.claude_limit_backoff import hold_from_failure

    at = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)
    hold = hold_from_failure(
        "You've hit your weekly limit · resets Nov 27, 3am (UTC)", at, default_sec=3600
    )
    assert hold is not None and hold.reset_stated
    assert hold.resets_at == datetime(2026, 11, 27, 3, 0, tzinfo=timezone.utc)


def test_long_error_naming_reset_late_still_reads_the_reset() -> None:
    """The failed brief's reason is clipped to 200 chars; the hold must read
    the unclipped exception text."""
    memory = ClaudeLimitMemory()
    clock = _Clock(datetime(2026, 11, 24, 9, 12, tzinfo=timezone.utc))
    claude = _Claude()
    claude.fail_with = "x" * 400 + " " + LIVE_TEXT_1124
    _hire(1, claude=claude, meter=_Meter(), memory=memory, clock=clock)
    hold = memory.active(clock.at)
    assert hold is not None and hold.reset_stated
    assert hold.resets_at == datetime(2026, 11, 24, 12, 10, tzinfo=timezone.utc)


def test_boot_seed_empty_outcome_clears() -> None:
    boot = datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)
    memory = ClaudeLimitMemory()
    rows = [
        {"written_at": _ms(boot - timedelta(hours=2)), "status": "failed",
         "refusal_reason": LIVE_TEXT_1124, "summary": ""},
        {"written_at": _ms(boot - timedelta(hours=1)), "status": "empty",
         "refusal_reason": None, "summary": ""},
    ]
    assert seed_claude_limit_memory(
        Settings(_env_file=None), graph_client=_Graph(rows), memory=memory, now=boot
    ) is None
    assert "'empty'" in RECENT_CLAUDE_OUTCOME_CYPHER


def test_default_module_memory_is_the_production_wiring() -> None:
    """No injected memory: the process-wide MEMORY carries the hold between
    hires (conftest resets it around each test)."""
    clock = _Clock(datetime(2026, 11, 24, 9, 12, tzinfo=timezone.utc))
    claude = _Claude()
    claude.fail_with = LIVE_TEXT_1124
    for n in (1, 2):
        handle_help_request(
            _help(n),
            settings=Settings(_env_file=None),
            claude=claude,
            observe_claude_limit=lambda: _Meter(),
            persist=lambda _b: None,
            now=clock,
        )
        clock.at = datetime(2026, 11, 24, 11, 23, tzinfo=timezone.utc)
    assert claude.calls == 1
