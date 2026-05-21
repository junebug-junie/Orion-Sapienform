from __future__ import annotations

import os
import sys

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

from app.settings import Settings  # noqa: E402


def test_daily_goal_archive_enabled_by_default() -> None:
    cfg = Settings(
        ACTIONS_DAILY_GOAL_ARCHIVE_ENABLED="true",
        ACTIONS_DAILY_GOAL_ARCHIVE_HOUR_LOCAL="3",
        ACTIONS_DAILY_GOAL_ARCHIVE_MINUTE_LOCAL="15",
    )
    assert cfg.actions_daily_goal_archive_enabled is True
    assert cfg.actions_daily_goal_archive_hour_local == 3
    assert cfg.actions_daily_goal_archive_minute_local == 15
