from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.scheduler_cursor_store import (
    SchedulerCursorStore,
    resolve_scheduler_cursor_store_path,
    scheduler_cursor_completed_local_date,
)


def test_resolve_scheduler_cursor_store_path_default_next_to_workflow(tmp_path: Path) -> None:
    wf = tmp_path / "nested" / "workflow_schedules.json"
    wf.parent.mkdir(parents=True, exist_ok=True)
    wf.write_text("{}")
    resolved = resolve_scheduler_cursor_store_path(None, workflow_schedule_store_path=str(wf))
    assert resolved == wf.parent / "scheduler_cursors.json"


def test_resolve_scheduler_cursor_store_path_explicit_file(tmp_path: Path) -> None:
    target = tmp_path / "cursors.json"
    resolved = resolve_scheduler_cursor_store_path(str(target), workflow_schedule_store_path="/ignored")
    assert resolved == target


def test_scheduler_cursor_store_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "scheduler_cursors.json"
    store = SchedulerCursorStore(path)
    store.set_last_completed("daily_pulse_v1", "2026-05-02")
    store.set_last_completed("world_pulse", "2026-05-02")
    store2 = SchedulerCursorStore(path)
    assert store2.get("daily_pulse_v1") == "2026-05-02"
    assert store2.get("world_pulse") == "2026-05-02"
    raw = json.loads(path.read_text())
    assert raw["daily_pulse_v1"] == "2026-05-02"


def test_scheduler_cursor_store_rejects_invalid_date(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"daily_pulse_v1": "not-a-date"}))
    store = SchedulerCursorStore(path)
    assert store.get("daily_pulse_v1") is None


def test_scheduler_cursor_completed_local_date_forced_uses_window() -> None:
    assert (
        scheduler_cursor_completed_local_date(
            forced_date="2026-04-30",
            window_request_date="2026-04-30",
            scheduled_local_date="2026-05-02",
        )
        == "2026-04-30"
    )


def test_scheduler_cursor_completed_local_date_normal_uses_scheduled() -> None:
    assert (
        scheduler_cursor_completed_local_date(
            forced_date=None,
            window_request_date="2026-05-01",
            scheduled_local_date="2026-05-02",
        )
        == "2026-05-02"
    )


def test_scheduler_cursor_completed_local_date_whitespace_forced_none() -> None:
    assert (
        scheduler_cursor_completed_local_date(
            forced_date="  ",
            window_request_date="2026-04-30",
            scheduled_local_date="2026-05-02",
        )
        == "2026-05-02"
    )
