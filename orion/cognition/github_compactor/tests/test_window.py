from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.cognition.chat_history_compactor.window import resolve_chat_compactor_window
from orion.cognition.github_compactor.window import resolve_github_compactor_window

NOW = datetime(2026, 9, 29, 12, 10, tzinfo=timezone.utc)  # 06:10 MDT, the scheduled slot


def test_scheduled_run_is_previous_denver_day_same_bounds_as_chat() -> None:
    gh = resolve_github_compactor_window(workflow_request={"scheduled_dispatch": {"x": 1}}, now=NOW, lookback_days=1)
    chat = resolve_chat_compactor_window(
        window_mode=None, lookback_hours=None, now=NOW, user_text="", workflow_request={"scheduled_dispatch": {"x": 1}}
    )
    assert gh.mode == "day"
    assert gh.window_label == gh.calendar_date == "2026-09-28"
    assert gh.window_start == chat.window_start == datetime(2026, 9, 28, 6, 0, tzinfo=timezone.utc)
    assert gh.window_end == chat.window_end


def test_on_demand_run_stays_rolling() -> None:
    gh = resolve_github_compactor_window(workflow_request={}, now=NOW, lookback_days=3)
    assert gh.mode == "rolling"
    assert gh.window_label == "2026-09-29"
    assert (gh.window_end - gh.window_start).days == 3


def test_explicit_window_mode_wins_and_bad_mode_rejected() -> None:
    assert resolve_github_compactor_window(workflow_request={"window_mode": "day"}, now=NOW, lookback_days=1).mode == "day"
    with pytest.raises(ValueError):
        resolve_github_compactor_window(workflow_request={"window_mode": "week"}, now=NOW, lookback_days=1)


def test_dst_transition_day_is_one_local_day() -> None:
    # 2026-11-01 is the US fall-back day in Denver (25 hours long).
    gh = resolve_github_compactor_window(
        workflow_request={"window_mode": "day"}, now=datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc), lookback_days=1
    )
    assert gh.calendar_date == "2026-11-01"
    assert round((gh.window_end - gh.window_start).total_seconds() / 3600) == 25
