from __future__ import annotations

from datetime import datetime, timezone

from app.main import _world_pulse_daily_dedupe_key, build_daily_window
from app.settings import Settings


def test_actions_world_pulse_disabled_by_default() -> None:
    cfg = Settings()
    assert cfg.actions_world_pulse_enabled is False


def test_world_pulse_scheduler_dedupe_key_is_deterministic() -> None:
    window = build_daily_window(
        now_utc=datetime(2026, 4, 26, 16, 0, tzinfo=timezone.utc),
        tz_name="America/Denver",
        override_date="2026-04-25",
    )
    key_a = _world_pulse_daily_dedupe_key(window)
    key_b = _world_pulse_daily_dedupe_key(window)
    assert key_a == key_b
    assert key_a.startswith("actions:world_pulse:daily:2026-04-25:")
