from __future__ import annotations

import logging
from datetime import datetime, timezone

from orion.world_pulse_read.timestamps import stamp_server_created_at

FIXED = datetime(2026, 10, 10, 22, 36, tzinfo=timezone.utc)


def test_model_created_at_is_replaced_and_logged(caplog) -> None:
    parsed = {"created_at": "2026-10-11T02:30:00Z"}
    with caplog.at_level(logging.INFO, logger="orion.world_pulse_read.timestamps"):
        stamped = stamp_server_created_at(parsed, seed_id="s1", stage="stage1", now=lambda: FIXED)
    assert stamped == FIXED
    assert parsed["created_at"] == FIXED.isoformat()
    assert "world_pulse_read_model_created_at_dropped" in caplog.text
    assert "2026-10-11T02:30:00Z" in caplog.text


def test_missing_created_at_is_stamped_without_log(caplog) -> None:
    parsed: dict = {}
    with caplog.at_level(logging.INFO, logger="orion.world_pulse_read.timestamps"):
        stamp_server_created_at(parsed, seed_id="s1", stage="stage2", now=lambda: FIXED)
    assert parsed["created_at"] == FIXED.isoformat()
    assert "dropped" not in caplog.text
