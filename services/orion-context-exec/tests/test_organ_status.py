"""Tests for per-run organ_status reporting."""

from __future__ import annotations

from app.organ_status import initial_organ_status, record_recall, record_trace
from orion.schemas.context_exec import ContextExecPermissionV1


def test_recall_timeout_records_attempted_failed() -> None:
    from app.settings import settings

    status = initial_organ_status(ContextExecPermissionV1(), settings)
    record_recall(
        status,
        {
            "hits": [],
            "error": "Timeout reading from 100.92.216.81:6379",
            "query": "do you recall where my mom lives?",
        },
    )
    recall = status["recall"]
    assert recall["enabled"] is True
    assert recall["attempted"] is True
    assert recall["ok"] is False
    assert recall["hit_count"] == 0
    assert "Timeout reading from" in str(recall["error"])


def test_trace_empty_success_marks_ok() -> None:
    from app.settings import settings

    status = initial_organ_status(ContextExecPermissionV1(), settings)
    record_trace(status, [])
    trace = status["trace"]
    assert trace["enabled"] is True
    assert trace["attempted"] is True
    assert trace["ok"] is True
    assert trace["hit_count"] == 0
    assert trace["error"] is None


def test_repo_disabled_stays_unattempted() -> None:
    from app.settings import settings

    status = initial_organ_status(ContextExecPermissionV1(), settings)
    assert status["repo"]["enabled"] is False
    assert status["repo"]["attempted"] is False
