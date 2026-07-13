from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from orion.cognition.chat_history_compactor.window import (
    ResolvedChatCompactorWindow,
    exclude_workflow_notification_turns,
    resolve_chat_compactor_window,
)
from orion.schemas.discussion_window import DiscussionWindowResultV1, DiscussionWindowTurnV1


def test_resolve_day_window_uses_yesterday_denver() -> None:
    now = datetime(2026, 7, 9, 6, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode="day",
        lookback_hours=None,
        now=now,
        user_text="",
        workflow_request={"window_mode": "day"},
    )
    assert isinstance(resolved, ResolvedChatCompactorWindow)
    assert resolved.mode == "day"
    assert resolved.calendar_date == "2026-07-08"
    assert resolved.compactor_index == "chat_compactor:day:2026-07-08"
    assert resolved.lookback_seconds == 86400
    assert resolved.window_end.astimezone(ZoneInfo("America/Denver")).date().isoformat() == "2026-07-08"


def test_resolve_rolling_from_prompt_six_hours() -> None:
    now = datetime(2026, 7, 9, 10, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode=None,
        lookback_hours=None,
        now=now,
        user_text="Compact the last 6 hours of chat into a memory digest.",
        workflow_request={},
    )
    assert resolved.mode == "rolling"
    assert resolved.lookback_hours == 6
    assert resolved.compactor_index.startswith("chat_compactor:rolling:6h:")


def test_day_window_covers_end_of_day_microseconds() -> None:
    now = datetime(2026, 7, 9, 6, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode="day",
        lookback_hours=None,
        now=now,
        user_text="",
        workflow_request={},
    )
    end_local = resolved.window_end.astimezone(ZoneInfo("America/Denver"))
    assert end_local.timetz().replace(tzinfo=None) == time.max


def test_request_lookback_hours_capped_at_fourteen_days() -> None:
    now = datetime(2026, 7, 9, 10, 0, tzinfo=ZoneInfo("America/Denver"))
    resolved = resolve_chat_compactor_window(
        window_mode=None,
        lookback_hours=None,
        now=now,
        user_text="",
        workflow_request={"lookback_hours": 720},
    )
    assert resolved.lookback_hours == 24 * 14
    assert resolved.lookback_seconds == 24 * 14 * 3600


def test_unknown_window_mode_fails_loud() -> None:
    now = datetime(2026, 7, 9, 10, 0, tzinfo=ZoneInfo("America/Denver"))
    with pytest.raises(ValueError, match="unsupported_chat_compactor_window_mode:daily"):
        resolve_chat_compactor_window(
            window_mode="daily",
            lookback_hours=None,
            now=now,
            user_text="",
            workflow_request={},
        )


def _turn(prompt: str, response: str, *, minute: int) -> DiscussionWindowTurnV1:
    return DiscussionWindowTurnV1(
        created_at=datetime(2026, 7, 13, 10, minute, tzinfo=timezone.utc),
        source="hub_http",
        correlation_id=f"corr-{minute}",
        prompt=prompt,
        response=response,
    )


def test_exclude_workflow_notification_turns_drops_workflow_trigger_round_trips() -> None:
    organic = _turn("How's the weather looking?", "Clear skies today.", minute=0)
    workflow_trigger = _turn(
        "Compact the last 6 hours of chat into a memory digest.",
        "Workflow: Chat History Compactor\nStatus: completed\nResult: Compacted 1 chat turn(s).\nPersisted: none\nScheduled: none",
        minute=5,
    )
    scheduled_ack = _turn(
        "Run your dream cycle tomorrow morning.",
        "Workflow request accepted: dream_cycle (tomorrow morning). notify_on=completion.",
        minute=10,
    )
    failed = _turn("Run a self review.", "Workflow failed before completion.", minute=15)
    window = DiscussionWindowResultV1(
        window_start_utc=datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 7, 13, 10, 20, tzinfo=timezone.utc),
        turn_count=4,
        turns=[organic, workflow_trigger, scheduled_ack, failed],
        transcript_text="stale placeholder",
    )

    filtered = exclude_workflow_notification_turns(window)

    assert filtered.turn_count == 1
    assert [t.prompt for t in filtered.turns] == ["How's the weather looking?"]
    assert "weather" in filtered.transcript_text.lower()
    assert "Workflow" not in filtered.transcript_text


def test_exclude_workflow_notification_turns_does_not_false_positive_on_organic_mentions() -> None:
    organic = _turn(
        "What do you think of my new workflow for tagging photos?",
        "Workflows like that scale well once you batch the tagging step.",
        minute=0,
    )
    window = DiscussionWindowResultV1(
        window_start_utc=datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 7, 13, 10, 5, tzinfo=timezone.utc),
        turn_count=1,
        turns=[organic],
        transcript_text="placeholder",
    )

    filtered = exclude_workflow_notification_turns(window)

    assert filtered.turn_count == 1
    assert filtered is window  # unchanged: no rebuild needed when nothing is dropped


def test_exclude_workflow_notification_turns_covers_schedule_management_and_chat_general_fallback() -> None:
    schedule_mgmt = _turn(
        "Cancel my dream cycle schedule.",
        "Workflow schedule cancel: ok\nCancelled schedule abc123.",
        minute=0,
    )
    chat_general_fallback = _turn(
        "Compact the last 24 hours of chat into a memory digest.",
        "Workflow 'chat_history_compactor_pass' failed and was not replaced with chat_general.",
        minute=5,
    )
    organic = _turn("Any thoughts on my sprint schedule?", "Looks reasonable to me.", minute=10)
    window = DiscussionWindowResultV1(
        window_start_utc=datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 7, 13, 10, 15, tzinfo=timezone.utc),
        turn_count=3,
        turns=[schedule_mgmt, chat_general_fallback, organic],
        transcript_text="placeholder",
    )

    filtered = exclude_workflow_notification_turns(window)

    assert filtered.turn_count == 1
    assert filtered.turns[0].prompt == "Any thoughts on my sprint schedule?"


def test_exclude_workflow_notification_turns_all_noise_yields_empty_window() -> None:
    workflow_trigger = _turn(
        "Compact the last 6 hours of chat into a memory digest.",
        "Workflow: Chat History Compactor\nStatus: completed\nResult: Compacted 1 chat turn(s).",
        minute=5,
    )
    window = DiscussionWindowResultV1(
        window_start_utc=datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 7, 13, 10, 10, tzinfo=timezone.utc),
        turn_count=1,
        turns=[workflow_trigger],
        transcript_text="Workflow: Chat History Compactor...",
    )

    filtered = exclude_workflow_notification_turns(window)

    assert filtered.turn_count == 0
    assert filtered.transcript_text == ""
