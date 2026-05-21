from __future__ import annotations

from unittest.mock import MagicMock, patch

from orion.autonomy.goal_archive import (
    build_archive_candidates,
    goal_archive_enabled,
    maybe_archive_after_goal_publish,
)


def test_build_archive_candidates_keeps_highest_priority_per_drive_origin() -> None:
    rows = [
        {
            "artifact_id": "g1",
            "drive_origin": "autonomy",
            "goal_statement_base": "Same text",
            "priority": 0.9,
            "proposal_status": "proposed",
        },
        {
            "artifact_id": "g2",
            "drive_origin": "autonomy",
            "goal_statement_base": "Same text",
            "priority": 0.3,
            "proposal_status": "proposed",
        },
        {
            "artifact_id": "g3",
            "drive_origin": "relational",
            "goal_statement_base": "Other",
            "priority": 0.5,
            "proposal_status": "proposed",
        },
    ]
    to_archive = build_archive_candidates(rows, max_active_per_subject=3, retention_days=30)
    assert set(to_archive) == {"g2"}


def test_goal_archive_enabled_false_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AUTONOMY_GOAL_ARCHIVE_ENABLED", raising=False)
    assert goal_archive_enabled() is False


def test_maybe_archive_after_goal_publish_skips_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTONOMY_GOAL_ARCHIVE_ENABLED", "false")
    with patch("orion.autonomy.goal_archive.archive_subject_goals") as mock_archive:
        maybe_archive_after_goal_publish(subject="orion")
        mock_archive.assert_not_called()


def test_maybe_archive_after_goal_publish_runs_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTONOMY_GOAL_ARCHIVE_ENABLED", "true")
    monkeypatch.setenv("AUTONOMY_GOAL_ARCHIVE_MAX_UPDATES_PER_TICK", "10")
    with patch("orion.autonomy.goal_archive.archive_subject_goals") as mock_archive:
        maybe_archive_after_goal_publish(subject="relationship")
        mock_archive.assert_called_once_with("relationship", dry_run=False, max_updates=10)
